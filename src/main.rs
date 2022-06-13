use nalgebra::{DVector, DMatrix};

use rand::{SeedableRng, Rng, prelude::SliceRandom, distributions::Slice};

use rbf_interp::{Basis, Scatter};


#[derive(Clone)]
struct RBFNetwork {
    //
    // Hidden layer
    //

    // Center point of RBF
    centers: DMatrix<f64>,
    // Width of each RBF center
    widths: DVector<f64>,

    //
    // Outputs
    //

    // Weights NUM_OUTPUTS * NUM_HIDDEN
    weights: DMatrix<f64>,
    // Bias of each output layer
    biases: DVector<f64>,

}


// How many times print information about training.
// Default: 10 mean that if you set 1000 epoch max,
// it will print message each 100 times.
// 0 - disable print
const NUM_INFO_PRINTS: usize = 6;

const LR: f64 = 0.005;

impl RBFNetwork {

    

    pub fn new(
        // R number generator
        generator: &mut rand::rngs::StdRng,
        // Count of centers in RBF network - number of neurons in hidden layer.
        number_of_centers: usize,
        // Count of output layers - number of output neourns.
        number_of_outputs: usize,
        test_data: &[(DVector<f64>, DVector<f64>)],
        // Width of neurons, used single value for all centers
        sigma: f64,

    ) -> Self {

        let items: Vec<_> = test_data.choose_multiple(generator,number_of_centers).collect();
        
        let n_cords = test_data[0].0.nrows();
        // RANDOM: Set centers
        let centers = DMatrix::from_fn(n_cords, number_of_centers,  |row, line| 
            items[line].0[row]
        );
        
        // setting width of each center = to single value "sigma"
        let widths = DVector::from_fn(number_of_centers, |_, _| sigma);

        // RANDOM: Init weights
        let hi: f64 = 0.01;
        let lo = -hi;
        let weights = DMatrix::from_fn( number_of_outputs, number_of_centers, |_, _| {
            generator.gen_range((lo..hi))
        });
        // set bias to 0
        let biases = DVector::from_fn(number_of_outputs, |_, _| {
            0.
        });

        let mut this = Self {
            weights,
            widths,
            centers,
            biases
        };


        assert_eq!(this.biases.nrows(), number_of_outputs);
        assert_eq!(this.weights.nrows(), number_of_outputs);

        assert_eq!(this.weights.ncols(), number_of_centers);
        
       
        this
    }

    pub fn train(&mut self,
        generator: &mut rand::rngs::StdRng,
        // Count of iterations to evaluate training 
        max_epochs: usize,
        mut test_data: Vec<(DVector<f64>, DVector<f64>)>,
        best_network: &mut BestNetwork,
        seed: u64,
    ) {
        for epoch in 0..max_epochs {

            //dbg!(&epoch);
            test_data.shuffle(generator);
            for (coords, results) in &test_data {
                // inlined fn compute_outputs for access to hidden_outputs
                let (output, hidden_output) = self.compute_outputs_inner(coords.clone());
            
                // Update weights
                for (mut weights_column, hidden_output) in self.weights.column_iter_mut()
                                .zip(hidden_output.row_iter())
                {
                    for ((mut weight, output), result) in weights_column.row_iter_mut()
                        .zip(output.row_iter())
                        .zip(results.row_iter())
                    {
                        let output: f64 = output.x;
                        let hidden_output: f64 = hidden_output.x;
                        let result: f64 = result.x;
                        let delta = -1. * LR * (output - result) *
                        hidden_output * (output * (1. - output));
                        weight.x += delta;
                    }
                }
                // Update bias
                for ((mut bias, output), result) in self.biases.row_iter_mut()
                        .zip(output.row_iter())
                        .zip(results.row_iter())
                {

                    let output: f64 = output.x;
                    let result: f64 = result.x;
                    let bias_delta: f64 = -1. * LR * (output - result) *
                        1. * (output * (1. - output));
                    bias.x += bias_delta;
                }
            }

            #[allow(unconditional_panic)] // rust will blame you for dividing by zero
            if NUM_INFO_PRINTS != 0 && epoch % (max_epochs / NUM_INFO_PRINTS) == 0 {
                println!("Epoch: {}", epoch);
                self.print_statis(&test_data);
            }

            let (accuracy, error) = self.statis(&test_data);
            if best_network.accuracy < accuracy {
                println!("Replacing best network with network: seed={}, epoch={}, accuracy={}", seed, epoch, accuracy);
                *best_network = 
                BestNetwork {
                    seed,
                    epoch,
                    accuracy,
                    network: self.clone()
                }
            }
        }
    }


    fn compute_outputs_inner(&self, coords: DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let n = self.centers.len();
        let hidden_output = DVector::from_fn(self.widths.nrows(), |row, _c| {
            let center_cord = &self.centers.column(row);
            let distance = (&coords - center_cord).norm();
            Self::rbf_gaussian(self.widths[row], distance)
        });
        let output = (&self.weights * &hidden_output);
        let biased_output =output + &self.biases;

        (( biased_output), hidden_output)
    }

    pub fn compute_outputs(&self, coords: DVector<f64>) -> DVector<f64> {
        self.compute_outputs_inner(coords).0
    }

    
    // public double Accuracy(double[][] dataX, int[][] dataY)
    // {
        
    //   double sumSE = 0.0;
    //   int nc = 0; int nw = 0;
    //   for (int i = 0; i < dataX.Length; ++i)
    //   {
    //     double[] X = dataX[i];  // inputs
    //     int[] Y = dataY[i];  // targets like (0, 1, 0)
    //     double[] oupt = this.ComputeOutputs(X);  // computeds like (0.2, 0.7, 0.1)
    //     int idx = ArgMax(oupt);  // location of largest prob
    // for (int k = 0; k < this.no; ++k)
    // {
    //   double err = Y[k] - oupt[k];  // target - output
    //   sumSE += err * err;  // (t - o)^2
    // }
    //     if (Y[idx] == 1)  // the corresponding target is 1
    //       ++nc;  // correct prediction
    //     else
    //       ++nw;  // wrong
    //   }
    //   return (nc * 1.0) / (nc + nw);
    // }

    // Return accuracy and error
    fn statis(&self, data: &[(DVector<f64>, DVector<f64>)]) -> (f64, f64) {
        let mut sum_se = 0.;
        let mut nc = 0.;
        let mut nw = 0.;
        for (coords, results) in data {
            let outputs = self.compute_outputs(coords.clone());

            for (output, result) in outputs.row_iter()
                .zip(results.row_iter()) {
                let err = result.x - output.x;
                sum_se += (err*err);
            }
            let idx = Self::max_predict_index(outputs).unwrap();
            if results[idx] == 1. {
                nc += 1.;
            } else {
                nw += 1.;
            }
        }
        let accuracy = (nc * 1.0) / (nc + nw);
        let error = sum_se;
        (accuracy, error)
        
    }

    fn print_statis(&self, data: &[(DVector<f64>, DVector<f64>)]) {
        let (accuracy, error): (f64, f64) = self.statis(data);
        println!("Error: {}", error);
        println!("Accuracy: {}", accuracy);
    }
    fn max_predict_index(result: DVector<f64>) -> Option<usize> {
        result.row_iter().enumerate().max_by(|(_, i1),(_, i2)|i1.x.partial_cmp(&i2.x).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0)
    }

    // Gaussian function that we used as radial basis in our neural network
    fn rbf_gaussian(width:f64, r: f64) -> f64 {
        (-(r / width).powi(2)).exp()
    }

    fn cmp_accuracy(&self, other: &Self, test_data: &[(DVector<f64>, DVector<f64>)]) -> std::cmp::Ordering {
        let our_stats = self.statis(test_data);
        let their_stats = other.statis(test_data);
        our_stats.0.partial_cmp(&their_stats.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}


// During training it is possible that network start to degradate, to stop this
struct BestNetwork {
    accuracy: f64,
    seed: u64,
    epoch: usize,
    network: RBFNetwork,
}

fn main() {
    
    let mut test_data_x:Vec<Vec<f64>> = vec![];
    test_data_x.push(vec![ 0.4, 0.6 ]);  // class 0
    test_data_x.push(vec![ 0.5, 0.6 ]);
    test_data_x.push(vec![ 0.6, 0.6 ]);
    test_data_x.push(vec![ 0.4, 0.5 ]);
    test_data_x.push(vec![ 0.5, 0.5 ]);
    test_data_x.push(vec![ 0.6, 0.5 ]);
    test_data_x.push(vec![ 0.5, 0.4 ]);
    test_data_x.push(vec![ 0.4, 0.4 ]);
    test_data_x.push(vec![ 0.6, 0.4 ]);

    test_data_x.push(vec![ 0.4, 0.7 ]);  // class 1
    test_data_x.push(vec![ 0.5, 0.7 ]);
    test_data_x.push(vec![ 0.6, 0.7 ]);
    test_data_x.push(vec![ 0.3, 0.6 ]);
    test_data_x.push(vec![ 0.7, 0.6 ]);
    test_data_x.push(vec![ 0.3, 0.5 ]);
    test_data_x.push(vec![ 0.7, 0.5 ]);
    test_data_x.push(vec![ 0.3, 0.4 ]);
    test_data_x.push(vec![ 0.7, 0.4 ]);

    test_data_x.push(vec![ 0.2, 0.6 ]);  // class 2
    test_data_x.push(vec![ 0.2, 0.5 ]);
    test_data_x.push(vec![ 0.2, 0.4 ]);
    test_data_x.push(vec![ 0.3, 0.3 ]);
    test_data_x.push(vec![ 0.4, 0.3 ]);
    test_data_x.push(vec![ 0.8, 0.6 ]);
    test_data_x.push(vec![ 0.8, 0.5 ]);
    test_data_x.push(vec![ 0.8, 0.4 ]);
    test_data_x.push(vec![ 0.7, 0.3 ]);
    test_data_x.push(vec![ 0.6, 0.3 ]);


    let mut test_data_y = vec![];
    test_data_y.push(vec![ 1, 0, 0 ]);  // class 0
    test_data_y.push(vec![ 1, 0, 0 ]);
    test_data_y.push(vec![ 1, 0, 0 ]);
    test_data_y.push(vec![ 1, 0, 0 ]);
    test_data_y.push(vec![ 1, 0, 0 ]);
    test_data_y.push(vec![ 1, 0, 0 ]);
    test_data_y.push(vec![ 1, 0, 0 ]);
    test_data_y.push(vec![ 1, 0, 0 ]);
    test_data_y.push(vec![ 1, 0, 0 ]);

    test_data_y.push(vec![ 0, 1, 0 ]);  // class 1
    test_data_y.push(vec![ 0, 1, 0 ]);
    test_data_y.push(vec![ 0, 1, 0 ]);
    test_data_y.push(vec![ 0, 1, 0 ]);
    test_data_y.push(vec![ 0, 1, 0 ]);
    test_data_y.push(vec![ 0, 1, 0 ]);
    test_data_y.push(vec![ 0, 1, 0 ]);
    test_data_y.push(vec![ 0, 1, 0 ]);
    test_data_y.push(vec![ 0, 1, 0 ]);

    test_data_y.push(vec![ 0, 0, 1 ]);  // class 2
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);
    test_data_y.push(vec![ 0, 0, 1 ]);


    let test_data: Vec<(DVector<f64>, DVector<f64>)> = test_data_x.into_iter().zip(test_data_y.into_iter())
    .map(|(x,y)|
        
        (DVector::from_vec(x),

        DVector::from_iterator(3, y.into_iter().map(|i| i as f64)))

    ).collect();


    let mut best_network = None;
    for seed in 0..50 {
        // let seed = 11;
        let mut generator = rand::rngs::StdRng::seed_from_u64(seed);
        println!("Creating a 2-15-3 RBF network, seed = {}", seed);
        let max_epochs = 1000;
        let sigma = 0.1;
        println!("Setting LR = {}, max_epochs = {}, sigma = {}", LR, max_epochs, sigma);
        println!("Starting training");
        let mut network = RBFNetwork::new( &mut generator, 14, 3, &test_data, sigma);
    
        let mut current_best = best_network.take().unwrap_or_else(||
        BestNetwork {
            seed,
            epoch: 0,
            accuracy: 0.,
            network: network.clone()
        });
        network.train(&mut generator, max_epochs, test_data.clone(), &mut current_best, seed);
        network.print_statis(&test_data);
        let unk = DVector::from_vec(vec![ 0.15, 0.25] );
        println!("Predicting class for: ");
    
        println!("input = {:?}", unk);
        let outputs = network.compute_outputs(unk);
        
        println!("output = {:?}", outputs);

        best_network = Some(current_best);
    }


    let best_network = best_network.unwrap();
    println!("The best network is on: seed={}, epoch={}", best_network.seed, best_network.epoch);
    best_network.network.print_statis(&test_data);
    
   



}
