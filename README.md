# Как работает

2 главных файла - 1. Создание графов из дфт 2. Обучение и тест
```Создание графов из дфт
python train_lgnn/dft2graphs_xyz.py
```
Чтобы команда сверху сработала положите данные в cheat\new_data_train\graphs_site_specific.

```Обучение и тест
Train on fives, test on pairs
python train_xyz.py --train fives --test pairs --adsorbate both

Train on triplets, test on all three types
python train_xyz.py --train triplets --test all --adsorbate H

Train on pairs, test on triplets
python train_xyz.py --train pairs --test triplets --adsorbate S
```
