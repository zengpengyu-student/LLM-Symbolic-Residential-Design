# LLM Discovery over Symbolic Representations Enables Compliant, High-Performance Residential Designs

This repository provides a release of the core generation and evaluation code associated with the paper **“LLM discovery over symbolic representations enables compliant, high-performance residential designs.”**

It is intended to document the main implementations used in the study and to provide representative inference and evaluation workflows.

## Overview

The framework explores residential designs through an editable symbolic representation and then instantiates and evaluates candidate layouts computationally.

The released modules cover:

1. **Coarse layout generation** using a conditional MeanFlow model with a Diffusion Transformer (DiT) backbone.
2. **Fine layout refinement** using a conditional denoising diffusion probabilistic model (DDPM) with a U-Net backbone.
3. **Deterministic compliance evaluation** using vector-derived geometry and jurisdiction-specific regulatory checks.
4. **Building energy simulation** using EnergyPlus.

The spatial-efficiency implementation is not included in this release; related methodological references are provided below.

## Repository structure

```text
.
├── Checkpoint/
│   ├── Coarse_layout_generation.pt
│   └── Fine_layout_generation.pth
│
├── Coarse_layout_generation/
│   ├── Inference.py
│   ├── meanflow.py
│   ├── CLIP_load.py
│   ├── models/
│   └── LICENSE
│
├── Fine_layout_generation/
│   ├── Inference.py
│   ├── predict_sample.py
│   ├── ddpm.py
│   ├── nets/
│   ├── vectorization.py
│   ├── figure_redraw.py
│   └── LICENSE
│
├── Compliance_check/
│   ├── run_standard_check.py
│   ├── toronto_compliance_checker.py
│   ├── vector_renderer.py
│   ├── FireTest2.py
│   └── config_from_Toronto.yaml
│
└── Energy_simulation/
    ├── run_energy_simulation.py
    └── engine/
```

## Model checkpoints

The pretrained coarse- and fine-generation checkpoints are not stored directly in this repository because of their file size.

Download them from:

**[Google Drive – pretrained checkpoints](https://drive.google.com/drive/folders/1iKpiidgWHnoAWR3zFbuuI429iGen6_Uo?usp=drive_link)**

After downloading, place the files in:

```text
Checkpoint/
├── Coarse_layout_generation.pt
└── Fine_layout_generation.pth
```

The coarse generator also uses the OpenCLIP ViT-H/14 text encoder. Its pretrained weights are loaded from the Hugging Face Hub at runtime:

```text
hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

An internet connection is therefore required the first time this model is loaded unless the Hugging Face files are already cached locally.

## Installation

The code was developed with Python 3.8-series environments. GPU inference requires a CUDA-compatible PyTorch installation.

The main Python dependencies include:

```text
torch
torchvision
numpy
opencv-python-headless
Pillow
timm
einops
open_clip_torch
natsort
svgwrite
PyYAML
pandas
eppy
```

For the energy-simulation module, **EnergyPlus 23.1** must also be installed separately.

Module-specific requirement files may be used to keep the generative, compliance, and EnergyPlus environments independent.

## 1. Coarse layout generation

`Coarse_layout_generation/` contains the conditional MeanFlow inference implementation.

The generator uses a DiT backbone conditioned on seven spatial condition maps representing:

- room-type information for living rooms;
- bedrooms;
- kitchens;
- bathrooms;
- balconies;
- the room-adjacency graph; and
- the valid site/buildable boundary.

The implementation also retains an OpenCLIP text-conditioning branch. In the compact inference example, the text condition is optional and an empty string is supported.

The principal entry point is:

```bash
cd Coarse_layout_generation
python Inference.py
```

Before running, provide the required spatial condition maps and adjust the input/output paths in the example script as needed.

The released inference path performs direct conditional Euler sampling from stochastic initialization.

## 2. Fine layout refinement

`Fine_layout_generation/` contains the source-conditioned DDPM used to refine coarse semantic layouts before vectorization.

The conditional U-Net receives the coarse layout together with the noisy target representation and performs iterative denoising. The refinement stage is intended to regularize semantic regions and boundaries while preserving the global spatial organization established by the coarse generator.

The main entry point is:

```bash
cd Fine_layout_generation
python Inference.py
```

The released implementation provides the standard ancestral DDPM inference path.

DDIM-style accelerated sampling can be incorporated by replacing the reverse-time sampling schedule with a DDIM sampler. A reference implementation is available from the original DDIM project:

**[ermongroup/ddim](https://github.com/ermongroup/ddim)**

The fine-generation workflow also contains raster-to-vector conversion utilities used to extract room and architectural-element geometries from refined semantic layouts.

## 3. Deterministic compliance evaluation

`Compliance_check/` contains a representative jurisdiction-specific deterministic checker using **Toronto** as the released example.

The authoritative input is a vector JSON representation of the floor plan.

The released checks include representative layout- and room-level geometric constraints such as:

- site-boundary containment;
- evacuation distance;
- room-count/programme consistency;
- adjacency/topological relations; and
- room dimensional constraints.

Evacuation distance is evaluated on walkable geometry using an eight-neighbour shortest-path calculation with Euclidean diagonal step cost.

Run the example checker with:

```bash
cd Compliance_check
python run_standard_check.py
```

`config_from_Toronto.yaml` contains the regulatory values used by this released Toronto example. Other jurisdictions use the same general evaluation structure with jurisdiction-specific schemas and, where required, additional checker logic.

## 4. Energy simulation

`Energy_simulation/` contains the vector-to-EnergyPlus preprocessing, EnergyPlus execution, and annual-energy aggregation workflow.

The implementation uses **EnergyPlus 23.1** and city-specific EPW weather files for:

- Sydney;
- Toronto;
- Shenzhen; and
- London.

The main entry point is:

```bash
cd Energy_simulation
python run_energy_simulation.py
```

Set the EnergyPlus installation directory through the `ENERGYPLUS_DIR` environment variable when needed.

For example on Windows:

```text
ENERGYPLUS_DIR=C:\EnergyPlusV23-1-0
```

The reported annual energy metric is:

```text
annual energy =
    heating load / 3.0
  + cooling load / 3.0
  + lighting electricity
```

where EnergyPlus heating and cooling IdealLoads outputs are converted from joules to kWh and divided by an assumed COP of 3.0. Lighting electricity is added directly.

This metric is reported in **annual kWh** and is **not an Energy Use Intensity (EUI)**; no floor-area normalization is applied in the production aggregation.

The provided runner demonstrates the city-specific simulation configurations on a user-supplied vector layout. In the benchmark study, each case was evaluated using the weather file associated with its corresponding jurisdiction.

## Spatial-efficiency evaluation

The spatial-efficiency code used in the study is not included in this compact repository release.

The formulation is related to prior work on effective space use and furniture feasibility:

> Weber, R. E., Mueller, C. & Reinhart, C.  
> *A hypergraph model shows the carbon reduction potential of effective space use in housing.*  
> Nature Communications 15, 8327 (2024).

The accompanying repository for that work is available at:

**[ramonweber/hypergraph](https://github.com/ramonweber/hypergraph)**

The spatial-efficiency analysis in the present study uses room-level furniture-feasibility concepts and aggregates functional use over interior residential spaces.

## Scope of this release

The repository is intended to expose the principal model architectures, inference procedures, geometric evaluation logic, and EnergyPlus workflow in a concise and inspectable form.

## Third-party software and attribution

Parts of the released implementations build on or interface with third-party open-source software, including:

- PyTorch;
- OpenCLIP;
- EnergyPlus and eppy;
- MeanFlow-related implementation components;
- DDPM/U-Net implementation components; and
- the DDIM reference implementation linked above.

Please retain the license files distributed with individual modules and consult the corresponding upstream projects for their licensing terms.

## Citation

If you use this code, please cite our paper once it is published.

```bibtex
@article{llm_symbolic_residential_design,
  title   = {LLM discovery over symbolic representations enables compliant, high‑performance residential designs},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>},
  doi     = {<doi>}
}
```

## License

The original code contributed in this repository is released under the MIT License.
See `LICENSE` file for details.

Note that third‑party dependencies (e.g., Open‑CLIP) are governed by their respective licenses.