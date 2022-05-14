# 😆 CARAR: Correlation-aware Review Aspect Recommender System
This is the MATLAB implemention of our paper:

> *T. Wei, T. W. S Chow and J. Ma, “Modeling Self-representation Label Correlations for Textual Aspects and Emojis Recommendation”, in IEEE Transactions on Neural Networks and Learning Systems, 2022.* [Paper link](https://doi.org/10.1109/TNNLS.2022.3171335)

<kbd>![CARAR](https://raw.githubusercontent.com/Joinn99/RepositoryResource/master/CARAR_Framework.svg)</kbd>


## File

The proposed Correlation-aware Review Aspect Recommender System (CARAR) includes three detailed steps. All corresponding codes are stored in `Model/`:

1. `Model/CARAR_C.m` : Self-representation correlation mapping.
2. `Model/CARAR_LF.m` : Latent factors updating.
3. `Model/CARAR_W.m` : Incorporaing additional information.

Each function above is called by `Model/CARAR.m` to formulate the full CARAR model.

## Datasets

- Yelp [^1]
- Beeradvocate [^2]
- HotelRec [^3]
- OpenRice [^4] Text / Emoji

All datasets are stored in `Data/` in `.mat` Matlab workspace files. Each file include:
- `D` (N * d): Additional information for each record.
- `E` (N * l): Review aspect labels for each record.
- `is` (N * 1): Item ID for each record.
- `iu` (N * 1): User ID for each record.
- `Label_E`  (Only available for OpenRice Text / OpenRice Emoji): Descriptions for each review aspect.

You can download the datasets we used at [Google Drive](https://drive.google.com/file/d/1--OJ0a_2bEQ9yy64NX2JmYf1vFDyV72Z/view?usp=sharing). Extract the downloaded zip file and put them in `Data/`.

## Usage

### Environment

- [MathWorks MATLAB](https://www.mathworks.com/products/matlab.html) R2019b or laber
  - [Parallel Computing Toolbox](https://ww2.mathworks.cn/help/parallel-computing/index.html) required
- This code is designed to run in a GPU environment and is not adapted for environments without a GPU.

### Running demos

Specify the dataset in `demo.m` and run the script. The best hyperparameters will be loaded automatically to run the demo. 

If you like to choose different hyperparameters, change them manually in `Model/CARAR.m`. 

## How to cite

```bibtex
@ARTICLE{9774027,
  author={Wei, Tianjun and Chow, Tommy W. S. and Ma, Jianghong},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Modeling Self-Representation Label Correlations for Textual Aspects and Emojis Recommendation}, 
  year={2022},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TNNLS.2022.3171335}
}
```

## References
[^1]: [Yelp Dataset](https://www.yelp.com/dataset)
[^2]: J. McAuley, J. Leskovec, and D. Jurafsky, “Learning attitudes and attributes from multi-aspect reviews,” in *Proc. 2012 IEEE 12th International Conference on Data Mining (ICDM)*, Dec. 2012, 994 pp. 1020–1025.
[^3]: D. Antognini and B. Faltings, “HotelRec: A novel very large-scale hotel recommendation dataset,” in *Proc. 12th Language Resources and Evaluation Conference (LREC)*, May 2020, pp. 4917–4923.
[^4]: [OpenRice Hong Kong](https://www.openrice.com/en/hongkong)


