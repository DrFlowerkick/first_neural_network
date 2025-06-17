// reading in mnist files

use anyhow::{Result, Context};
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use std::fs::File;
use linfa::prelude::*;

const PIXEL_MAX: f64 = 255.0;
const INPUT_MIN: f64 = 0.01;
const INPUT_MAX: f64 = 1.0;

// normalize a pixel value from (0–255) to 0.01–1.0
fn normalize_pixel(pixel: u8) -> f64 {
    (pixel as f64 / PIXEL_MAX) * (INPUT_MAX - INPUT_MIN) + INPUT_MIN
}

// generate target array
fn target_array(label: u8) -> Array1<f64> {
    let mut arr = vec![0.01; 10];
    arr[label as usize] = 0.99;
    Array1::from(arr)
}

// returns vec of (input, target)
pub fn load_mnist_csv(path: &str) -> Result<Vec<(Array1<f64>, Array1<f64>)>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut dataset = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let label: u8 = record[0].parse()?;
        let pixels: Vec<f64> = record
            .iter()
            .skip(1)
            .map(|v| v.parse::<u8>().map(normalize_pixel).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        let input = Array1::from(pixels);
        let target = target_array(label);

        dataset.push((input, target));
    }

    Ok(dataset)
}

// Load MNIST-style CSV (label,pixel0,...,pixel783) for linfa
pub fn load_mnist_dataset(path: &str) -> Result<DatasetBase<Array2<f64>, Array1<usize>>> {
    let file = File::open(path).context("Failed to open CSV file")?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let mut features: Vec<f64> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for result in rdr.records() {
        let record = result.context("Error reading CSV row")?;
        let label: usize = record[0].parse().context("Invalid label")?;
        labels.push(label);

        for val in record.iter().skip(1) {
            let pixel = val.parse::<u8>().map(normalize_pixel).context("Invalid pixel")?;
            features.push(pixel);
        }
    }

    let n_samples = labels.len();
    let x: Array2<f64> = Array2::from_shape_vec((n_samples, 784), features)?;
    let y = Array1::from(labels);

    Ok(DatasetBase::<Array2<f64>, Array1<usize>>::new(x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_load_mnist_csv() {
        let test_file = "tests/mnist_code_test.csv";

        // Make sure the test file exists
        assert!(Path::new(test_file).exists(), "Test CSV file not found.");

        // Load dataset
        let data = load_mnist_csv(test_file).expect("Failed to load test MNIST data");

        // There should be exactly 1 sample
        assert_eq!(data.len(), 1);

        let (input, target) = &data[0];

        // Input should have 784 pixels (28x28)
        assert_eq!(input.len(), 784);

        // All normalized pixel values should be in range [0.01, 1.0]
        assert!(input.iter().all(|&v| v >= 0.01 && v <= 1.0));

        // Target vector should have 10 classes
        assert_eq!(target.len(), 10);

        // There should be exactly one 0.99 value in the target vector
        let high_vals: Vec<_> = target
            .iter()
            .filter(|&&v| (v - 0.99).abs() < 1e-6)
            .collect();
        assert_eq!(high_vals.len(), 1);

        // The index of 0.99 should match the label (expected: 7)
        let expected_label = 7;
        assert!((target[expected_label] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_load_mnist_dataset_from_file() {
        let path = Path::new("tests/mnist_code_test.csv");

        assert!(path.exists(), "Test CSV file not found");

        let dataset = load_mnist_dataset(path.to_str().unwrap())
            .expect("Failed to load MNIST test data");

        // Basic shape checks
        assert_eq!(dataset.nfeatures(), 784);
        assert!(dataset.nsamples() == 1);

        // Check that labels are in range [0, 9]
        for &label in dataset.targets().iter() {
            assert!(
                (0..=9).contains(&label),
                "Label out of range: {}",
                label
            );
        }

        // Check that all pixel values are normalized to [0.01, 1.0]
        for &val in dataset.records().iter() {
            assert!(
                (0.01..=1.0).contains(&val),
                "Pixel not normalized: {}",
                val
            );
        }
    }
}
