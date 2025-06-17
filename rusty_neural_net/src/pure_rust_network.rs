// creating neural network without framework

use crate::utils::sigmoid;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

// simple data structure to hold network weights
pub struct NeuralNetwork {
    pub w_input_hidden: Array2<f64>,
    pub w_hidden_output: Array2<f64>,
    pub learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::rng();

        let w_input_hidden =
            Array2::from_shape_fn((hidden_nodes, input_nodes), |_| rng.random_range(-0.5..0.5));

        let w_hidden_output = Array2::from_shape_fn((output_nodes, hidden_nodes), |_| {
            rng.random_range(-0.5..0.5)
        });

        Self {
            w_input_hidden,
            w_hidden_output,
            learning_rate,
        }
    }

    pub fn query(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let hidden_inputs = self.w_input_hidden.dot(inputs);
        let hidden_outputs = hidden_inputs.mapv(sigmoid);

        let final_inputs = self.w_hidden_output.dot(&hidden_outputs);
        // final outputs
        final_inputs.mapv(sigmoid)
    }

    pub fn train(&mut self, inputs: &Array1<f64>, targets: &Array1<f64>) {
        // forward
        let hidden_inputs = self.w_input_hidden.dot(inputs);
        let hidden_outputs = hidden_inputs.mapv(sigmoid);

        let final_inputs = self.w_hidden_output.dot(&hidden_outputs);
        let final_outputs = final_inputs.mapv(sigmoid);

        // calculate the error
        let output_errors = targets - &final_outputs;
        let hidden_errors = self.w_hidden_output.t().dot(&output_errors);

        // gradients + update
        let delta_output = &output_errors * &final_outputs * &(1.0 - &final_outputs);
        let delta_hidden = &hidden_errors * &hidden_outputs * &(1.0 - &hidden_outputs);

        // update weights as row vectors, because this is equal to transposed column vector, which we need
        // in .dot() to update the weights
        let hidden_outputs_2d = hidden_outputs.view().insert_axis(Axis(0));
        let inputs_2d = inputs.view().insert_axis(Axis(0));

        self.w_hidden_output = &self.w_hidden_output
            + self.learning_rate
                * delta_output
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&hidden_outputs_2d);
        self.w_input_hidden = &self.w_input_hidden
            + self.learning_rate * delta_hidden.view().insert_axis(Axis(1)).dot(&inputs_2d);
    }
}
