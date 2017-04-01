from experiments.run_mnist_experiment import (run)

run(
        [ 100 ],
        learning_rate=.1,
        corruption_rate=0,
        decay_rate=0,
        hidden_epochs=1000,
        seed=0,
        init_with='random',
    )
