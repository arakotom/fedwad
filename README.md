## This is the code repo for the paper Federated Wassertein Distance, appeared at [ICLR2024]

The paper is available at [https://openreview.net/forum?id=rsg1mvUahT](https://openreview.net/forum?id=rsg1mvUahT)


**Authors** : Alain Rakotomamonjy, Kimia Nadjahi, Liva Ralaivola

### Structure of the repository
* FedWad.py : contains the implementation of the proposed method
* otdd_script.py and otdd_expe.py  contain the codes for reproducing Figure 6 of the paper 
* fed_avg** : contains the code for reproducing the boosting FL experiments of the paper. 
    * entry point is the fedavg_script.py. It launches the experiments for the different datasets and clustering methods.
    * fedavg_expe.py contains the code for the experiments
    * fedavg_results.py contains the code for creating the lines of the reslt tables 
  

OTDD code comes from this repo. https://github.com/microsoft/otdd and FedRep code comes from the repo of Collins et al. 



If you use this code for your research, you can cite our paper and the POT library:

```
@inproceedings{rakoto2024fedwad,
  title={Federated Wassertein Distance},
  author={Rakotomamonjy, Alain and Nadjahi, Kimia and Ralaivola, Liva},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```


```
@article{flamary2021pot,
  title={Pot: Python optimal transport},
  author={Flamary, R{\'e}mi and Courty, Nicolas and Gramfort, Alexandre and Alaya, Mokhtar Z and Boisbunon, Aur{\'e}lie and Chambon, Stanislas and Chapel, Laetitia and Corenflos, Adrien and Fatras, Kilian and Fournier, Nemo and others},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={78},
  pages={1--8},
  year={2021}
}
```