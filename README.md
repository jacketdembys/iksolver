# Inverse Kinematics of Robotic Manipulators Using a New Learning-by-Example Method


## <div align="center">Requirements</div>
- pytorch:2.0.1
- cuda11.7
- cudnn8
- wandb
- scikit_learn
- numpy
- scipy
- pandas
- matplotlib
- tqdm


## <div align="center">Usage</div>

</details>
<details open><summary>Clone repository</summary>

```shell
git clone https://github.com/jacketdembys/iksolver.git
cd iksolver
```

</details>



</details>
<details open><summary>Generate datasets (ToDo)</summary>
</details>

</details>
<details open><summary>Train IK model</summary>
Choose/set the training configurations in the create_experiments.py file, then create a train.yaml configuration file:

```shell
python create_experiments.py
```

Run the training script to train/eval/test the model:

```shell
python ik-solver.py --config-path train.yaml
```

</details>

</details>
<details open><summary>Test IK model (ToDo)</summary>
</details>