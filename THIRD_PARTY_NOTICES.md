# Third-Party Notices

This repository includes, adapts, depends on, or references third-party software and data.  
The notices below are provided for attribution and to help users identify the applicable upstream licenses and terms.

## 1. Coarse-layout generation components

Parts of the coarse-layout generation implementation are distributed under the MIT License retained in:

`Coarse_layout_generation/LICENSE`

Copyright notice retained in that file:

> Copyright (c) 2025 Jiarui Hai

The original license file must remain with redistributed copies or substantial portions of the corresponding code.

## 2. Fine-layout generation components

Parts of the fine-layout generation implementation are distributed under the MIT License retained in:

`Fine_layout_generation/LICENSE`

Copyright notice retained in that file:

> Copyright (c) 2022 Bubbliiiing

The original license file must remain with redistributed copies or substantial portions of the corresponding code.

## 3. OpenCLIP

The coarse-layout module uses OpenCLIP and loads the pretrained model

`hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

from the Hugging Face Hub at runtime.

OpenCLIP and the pretrained model are third-party components and are subject to their respective upstream licenses and model terms. They are not relicensed by this repository.

## 4. EnergyPlus and eppy

The energy-simulation workflow interfaces with:

- EnergyPlus 23.1
- eppy 0.5-series

EnergyPlus is installed separately by the user and is not redistributed as an executable in this repository.  
eppy is installed as a Python dependency.

Both projects remain subject to their respective upstream licenses and terms.

## 5. DDIM reference implementation

This compact release includes the ancestral DDPM inference path but does not bundle a DDIM implementation.

For DDIM-style accelerated sampling, the README points to the original reference implementation:

https://github.com/ermongroup/ddim

No code from that repository is redistributed here unless otherwise stated in the source files.

## 6. Spatial-efficiency reference

The README cites the following prior work as a methodological reference for effective space use and furniture-feasibility concepts:

R. E. Weber, C. Mueller, and C. Reinhart,  
“A hypergraph model shows the carbon reduction potential of effective space use in housing,”  
Nature Communications 15, 8327 (2024).

Associated public repository:

https://github.com/ramonweber/hypergraph

The spatial-efficiency implementation from that repository is not bundled in this compact release.

## 7. Weather data

The EnergyPlus workflow uses EPW weather files originating from external weather-data providers.  
The original EPW headers and source notices must be preserved.

The bundled files contain source-specific notices, including, for example:

- London: IWEC data with an ASHRAE copyright/license notice.
- Sydney: Australian Reference Meteorological Year data with a Commonwealth of Australia / ACADS-BSG notice.
- Shenzhen: SWERA/NREL-derived weather data.
- Toronto: weather data identifying its original meteorological source and distribution source.

Weather-data redistribution rights are source-specific and are not granted by this repository. Before public release, the repository maintainer should confirm that redistribution of each bundled EPW file is permitted under its original terms. If redistribution is not permitted or is uncertain, replace the corresponding file with download/source instructions rather than redistributing the data itself.

## 8. General notice

Third-party names, software, pretrained models, and datasets remain the property of their respective owners.

Nothing in this repository supersedes the license, copyright notice, terms of use, or attribution requirements supplied by an upstream project or data provider.
