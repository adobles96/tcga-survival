from concurrent.futures import ThreadPoolExecutor
import os

import gigapath.slide_encoder as slide_encoder
import numpy as np
import pandas as pd
import openslide
import timm
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import data_utils

MASK_LEVEL = 2  # level to generate tissue mask
OCCUPANCY_THRESHOLD = 0.4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WSIProcessor:
    def __init__(self, tile_size=1024, overlap=0, max_tiles=None, tile_level=0):
        """
        Initialize WSI processor

        Args:
            tile_size: Size of tiles to extract at the given tile_level. E.g. 1024 at level 0 covers
                equivalent area to 256 at level 2.
            overlap: Overlap between tiles in pixels
            max_tiles: Maximum number of tiles to extract per WSI, None for all
            tile_level: Level to extract tiles from
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_tiles = max_tiles
        self.tile_level = tile_level
        self.tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.tile_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.slide_encoder = slide_encoder.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
        )
        self.tile_encoder.eval()
        self.slide_encoder.eval()
        self.tile_encoder.to(device)
        self.slide_encoder.to(device)

    def read_wsi(self, file_path):
        """Read WSI file using OpenSlide"""
        try:
            slide = openslide.OpenSlide(file_path)
            return slide
        except Exception as e:
            print(f"Error reading slide {file_path}: {str(e)}")
            return None

    def get_tissue_mask(self, slide, level=2):
        """ Generate tissue mask at given level. Uses Otsu thresholding on grayscale image """
        thumb = slide.get_thumbnail(
            (slide.dimensions[0] // 2**level, slide.dimensions[1] // 2**level)
        )
        thumb_gray = thumb.convert('L')
        thumb_array = np.array(thumb_gray)

        # Otsu thresholding
        thresh = np.mean(thumb_array)
        tissue_mask = thumb_array < thresh
        return tissue_mask

    def find_tissue_tiles(self, slide, level):
        """Find coordinates of tiles containing tissue"""
        # Generate tissue mask
        tissue_mask = self.get_tissue_mask(slide, level=level)

        tile_coords = []
        # Calculate tile size at given level
        scaled_tile_size = self.tile_size // 2**(level - self.tile_level)

        for y in range(0, tissue_mask.shape[0], scaled_tile_size):
            for x in range(0, tissue_mask.shape[1], scaled_tile_size):
                # Extract tile region from mask
                tile_mask = tissue_mask[y: y + scaled_tile_size, x: x + scaled_tile_size]

                # Check if tile contains sufficient tissue
                if tile_mask.size > 0 and np.mean(tile_mask) > OCCUPANCY_THRESHOLD:
                    # Coordinates are always in level 0 reference frame
                    orig_x = x * 2**level
                    orig_y = y * 2**level
                    tile_coords.append((orig_x, orig_y))

        return tile_coords

    def extract_tile(self, slide, x, y):
        """Extract a single tile from the WSI"""
        tile = slide.read_region((x, y), self.tile_level, (self.tile_size, self.tile_size))
        tile = tile.convert('RGB')
        return tile

    def embed_tile(self, tile: torch.tensor):
        """Embed a tile or batch of tiles using the tile encoder"""
        if tile.ndim == 3:
            tile = tile.unsqueeze(0)
        tile_tensor = self.tile_transforms(tile)
        with torch.no_grad():
            tile_embedding = self.tile_encoder(tile_tensor).squeeze()
        return tile_embedding

    def embed_wsi(self, file_path) -> torch.tensor:
        """ Generate embedding for a WSI """
        # Read WSI
        slide = self.read_wsi(file_path)
        if slide is None:
            return

        # Find tiles containing tissue
        tile_coords = self.find_tissue_tiles(slide, level=MASK_LEVEL)

        # Select subset of tiles if needed
        if self.max_tiles is not None and len(tile_coords) > self.max_tiles:
            chosen_indices = np.random.choice(
                np.arange(len(tile_coords)), self.max_tiles, replace=False
            )
            tile_coords = [tile_coords[i] for i in chosen_indices]

        # Extract tiles in parallel
        def get_tile(coords):
            return self.extract_tile(slide, *coords)

        with ThreadPoolExecutor() as executor:
            tiles = list(executor.map(get_tile, tile_coords))

        # Embed tiles
        tiles = torch.stack([to_tensor(t) for t in tiles], dim=0)
        tile_embeddings = self.embed_tile(self.tile_transforms(tiles).to(device))

        # Embed slide
        with torch.no_grad():
            # slide encoder expects input of shape (batch, n_tiles, d)
            return self.slide_encoder(
                tile_embeddings.unsqueeze(0).to(device), torch.tensor(tile_coords).to(device)
            )[0]   # returns single element list


# Example usage
# processor = WSIProcessor(tile_size=1024, max_tiles=16)
# processor.process_wsi("path/to/slide.sas")

# download and preprocess all slides to produce a dict of case_id to embedding
if __name__ == '__main__':
    cases = pd.read_csv("data/case_data.csv")
    processor = WSIProcessor(tile_size=256, max_tiles=2048, tile_level=2)
    for case_id in tqdm(cases["case_id"], unit='cases', colour='green'):
        if not os.path.exists(f"data/wsi_embeddings/{case_id}.pt"):
            diagnostic_slides = list(
                filter(
                    lambda x: (x['data_format'] == 'SVS') and
                    (x['experimental_strategy'] == 'Diagnostic Slide'),
                    data_utils.get_case_files(case_id)
                )
            )
            if len(diagnostic_slides) == 0:
                print(f"No diagnostic slides found for {case_id}")
                continue
            # if multiple slides use the first
            try:
                wsi_path = data_utils.download_file(
                    diagnostic_slides[0]['file_id'], "svs", "data",
                    diagnostic_slides[0]['file_size']
                )
            except Exception:
                print('Download failed!')
                continue
            # process slide
            embedding = processor.embed_wsi(wsi_path).cpu()
            # save embedding
            torch.save(embedding, f"data/wsi_embeddings/{case_id}.pt")
            os.remove(wsi_path)
