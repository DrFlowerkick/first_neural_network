// utils for neural networks

use ndarray::Array1;
use anyhow::{Result, Context};

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn argmax(arr: &Array1<f64>) -> Result<usize> {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .context("Expected non empty result vector.")
}
