// analyze pictures of numbers with linfa, not using a neural network

use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use anyhow::Result;
use rusty_neural_net::mnist::load_mnist_dataset;
use ndarray::{Array2, Array1};

fn main() -> Result<()> {
    // Load training and test data
    let full_train = load_mnist_dataset("../downloads/mnist_train.csv")?;
    let full_test = load_mnist_dataset("../downloads/mnist_test.csv")?;

    // Only use labels 0 and 1
    let train = filter_binary_classes(&full_train, &[0, 1]);
    let test = filter_binary_classes(&full_test, &[0, 1]);

    println!("Training on {} samples...", train.nsamples());

    // Train logistic regression model (multiclass)
    let model = LogisticRegression::default()
        .max_iterations(100)
        .fit(&train)?;

    println!("Model trained.");

    // Predict test set
    let pred = model.predict(&test);

    // Evaluate accuracy
    let cm = pred.confusion_matrix(&test)?;
    println!("Test accuracy: {:.2}%", cm.accuracy() * 100.0);

    Ok(())
}

fn filter_binary_classes(
    dataset: &DatasetBase<Array2<f64>, Array1<usize>>,
    allowed: &[usize],
) -> DatasetBase<Array2<f64>, Array1<usize>> {
    let mask: Vec<bool> = dataset
        .targets()
        .iter()
        .map(|label| allowed.contains(label))
        .collect();

    let records = dataset
        .records()
        .outer_iter()
        .zip(&mask)
        .filter_map(|(row, &keep)| if keep { Some(row.to_owned()) } else { None })
        .collect::<Vec<_>>();

    let targets = dataset
        .targets()
        .iter()
        .zip(&mask)
        .filter_map(|(&label, &keep)| if keep { Some(label) } else { None })
        .collect::<Vec<_>>();

    let flat_data: Vec<f64> = records.into_iter().flat_map(|row| row.to_vec()).collect();

    let x = Array2::from_shape_vec((targets.len(), dataset.nfeatures()), flat_data)
        .expect("Failed to reshape filtered features");

    let y = Array1::from(targets);

    DatasetBase::new(x, y)
}
