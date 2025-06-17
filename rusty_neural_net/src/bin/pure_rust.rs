// pure rusty network

use rusty_neural_net::{mnist::load_mnist_csv, pure_rust_network::NeuralNetwork, utils::argmax};
use ndarray::Array1;
use anyhow::Result;
use rand::{rng, seq::SliceRandom};

fn main() -> Result<()> {
    // Load training and test data
    let mut dataset = load_mnist_csv("../downloads/mnist_train.csv")?;
    let test_data = load_mnist_csv("../downloads/mnist_test.csv")?;

    // Create the network: 784 inputs, 100 hidden, 10 outputs
    let mut nn = NeuralNetwork::new(784, 100, 10, 0.1);

    println!("Starting training.");
    // Train for multiple epochs
    let epochs = 5;

    let mut rng = rng();
    let mut progress_counter: usize = 0;

    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);

        // shuffle dataset
        dataset.shuffle(&mut rng);

        for (input, target) in &dataset {
            nn.train(input, target);
            progress_counter += 1;
            if progress_counter % 10_000 == 0 {
                println!("Trained samples {}", progress_counter);
            }
        }

        println!("Finished epoch {}", epoch + 1);
    }

    println!("Training completed.");

    // Evaluate on test set
    let accuracy = evaluate_accuracy(&nn, &test_data)?;
    println!("Test accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

fn evaluate_accuracy(network: &NeuralNetwork, test_data: &[(Array1<f64>, Array1<f64>)]) -> Result<f64> {
    let mut correct = 0;

    for (input, target) in test_data {
        let output = network.query(input);
        let predicted = argmax(&output)?;
        let actual = argmax(target)?;

        if predicted == actual {
            correct += 1;
        }
    }

    Ok(correct as f64 / test_data.len() as f64)
}

