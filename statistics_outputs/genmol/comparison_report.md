# GenMol V2 ordering comparison

## De novo generation

| method                    |   diversity |   quality |   uniqueness |   validity |
|:--------------------------|------------:|----------:|-------------:|-----------:|
| DPRM-confidence-GenMol-V2 |      0.8103 |    0.6032 |       0.3129 |     0.9680 |
| DPRM-random-GenMol-V2     |      0.7778 |    0.8293 |       0.3643 |     0.9970 |
| GenMol-V2                 |      0.8285 |    0.8541 |       0.5822 |     0.9840 |
| Progressive-GenMol-V2     |      0.8533 |    0.5763 |       0.4798 |     0.9900 |


## Fragment-constrained generation

### validity

| method                    |   linker_design |   linker_design_onestep |   motif_extension |   scaffold_decoration |   superstructure_generation |
|:--------------------------|----------------:|------------------------:|------------------:|----------------------:|----------------------------:|
| DPRM-confidence-GenMol-V2 |          0.1418 |                  0.0000 |            1.0000 |                1.0000 |                      1.0000 |
| DPRM-random-GenMol-V2     |          0.4286 |                  0.5728 |            0.7180 |                1.0000 |                      1.0000 |
| GenMol-V2                 |          0.1418 |                  0.4304 |            0.7148 |                0.7165 |                      1.0000 |
| Progressive-GenMol-V2     |          0.1418 |                  0.0000 |            1.0000 |                1.0000 |                      1.0000 |


### quality

| method                    |   linker_design |   linker_design_onestep |   motif_extension |   scaffold_decoration |   superstructure_generation |
|:--------------------------|----------------:|------------------------:|------------------:|----------------------:|----------------------------:|
| DPRM-confidence-GenMol-V2 |          0.1418 |                  0.0000 |            0.4207 |                0.7124 |                      0.7097 |
| DPRM-random-GenMol-V2     |          0.2864 |                  0.2863 |            0.2829 |                0.7124 |                      0.7097 |
| GenMol-V2                 |          0.1418 |                  0.1439 |            0.2797 |                0.4289 |                      0.7097 |
| Progressive-GenMol-V2     |          0.1418 |                  0.0000 |            0.4207 |                0.7124 |                      0.7097 |


### diversity

| method                    |   linker_design |   linker_design_onestep |   motif_extension |   scaffold_decoration |   superstructure_generation |
|:--------------------------|----------------:|------------------------:|------------------:|----------------------:|----------------------------:|
| DPRM-confidence-GenMol-V2 |          0.0000 |                  0.0000 |            0.0000 |                0.0000 |                      0.0000 |
| DPRM-random-GenMol-V2     |          0.0000 |                  0.0000 |            0.0000 |                0.0000 |                      0.0000 |
| GenMol-V2                 |          0.0000 |                  0.0000 |            0.0000 |                0.0000 |                      0.0000 |
| Progressive-GenMol-V2     |          0.0000 |                  0.0000 |            0.0000 |                0.0000 |                      0.0000 |


### distance

| method                    |   linker_design |   linker_design_onestep |   motif_extension |   scaffold_decoration |   superstructure_generation |
|:--------------------------|----------------:|------------------------:|------------------:|----------------------:|----------------------------:|
| DPRM-confidence-GenMol-V2 |          0.3939 |                nan      |            0.6784 |                0.6742 |                      0.6746 |
| DPRM-random-GenMol-V2     |          0.4523 |                  0.3990 |            0.6430 |                0.6742 |                      0.6746 |
| GenMol-V2                 |          0.3939 |                  0.4192 |            0.6506 |                0.6028 |                      0.6746 |
| Progressive-GenMol-V2     |          0.3939 |                nan      |            0.6784 |                0.6742 |                      0.6746 |

