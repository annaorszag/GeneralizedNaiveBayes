# GeneralizedNaiveBayes

## Article

Generalized Naive Bayes by Edith Alice Kovács, Anna Ország, Dániel Pfeifer, András Benczúr:
https://arxiv.org/html/2408.15923v1

## How to run

1. Download the `src` folder.
2. Take note of your data file (in this example, it is `Wdbc/wdbc.csv`)
3. Run the code with `python src/run_gnb.py --data_path="Wdbc/wdbc.csv" --algorithm="GNBO" --num_of_intervals=5` (feel free to change the parameters).

## Parameters

- `data_path` the path to the data used
- `algorithm` is the algorithm used, either `GNBA` or `GNBO` (default is `GNBO`)
- `num_of_intervals` is the number of discretization intervals (default is `5`)
