# MCMRC

### Dependencies
You'll need to install transformers, torch, wandb, scipy

### Training
To train a standard Multiple Choice Reading Comprehension system, run the command

```
python run_train.py --path 'path_to_save_model' --data-set race --formatting standard
```

- ```--formatting standard``` sets the model to use full context inputs. To train the no context shortcut model, use the argument ```--formatting QO```
- ```--data-set race``` sets the dataset to used in training. To use other datasets load your own datasets and interface the code with src/data_utils/data_loader
- other training arguments can be modified, look into run_train.py to see what extra arguments can be used.


### Evaluation

To evaluate a trained model, use the command

```
python evaluation.py --path 'path_to_save_model' --data-set race
```
- ```--path 'path_to_save_model'``` is the path to the model to be evaluated. Note this should be the parent path (the same path argument used in training) and may have multiple seeds present
- ```--data-set race``` is the data set to be evaluated. Again, other datasets can be used if they are loaded into the data_utils. 
- ```--mode dev``` is an optional argument to look at the evaluation performance on the dev set. The default is ```test```

Note that evaluation caches results, and so the second time the same evaluation runs the results are generated near instantaneously. However these means that stale evaluations will be saved

