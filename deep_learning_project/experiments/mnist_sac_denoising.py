from experiments.run_mnist_experiment import (run)

run(
        [ 100, 50, 25, 10 ],
        learning_rate=.1,
        corruption_rate=.3,
        decay_rate=0,
        hidden_epochs=10,
        seed=0,
        init_with='sac',
    )
