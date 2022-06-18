use std::{fs::File, env, path::{PathBuf, Path}, io::{BufReader, BufRead}, collections::{HashSet, HashMap}, borrow::{Cow, Borrow}, process::exit};

use nalgebra::{DVector, DMatrix};

use rand::{SeedableRng, Rng, prelude::SliceRandom, distributions::Slice};

use clap::Parser;
use log::*;

pub mod kdd_schema;

use kdd_schema::{KDDRecordWithLable, KDDRecord};
use flate2::read::GzDecoder;

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
const MAX_STATS_ITEMS: usize = 100000;

impl RBFNetwork {

    

    pub fn new(
        // R number generator
        generator: &mut rand::rngs::StdRng,
        // Count of centers in RBF network - number of neurons in hidden layer.
        number_of_centers: usize,
        test_data: &[(DVector<f64>, DVector<f64>)],
        // Width of neurons, used single value for all centers
        sigma: f64,

    ) -> Self {

        let items: Vec<_> = test_data.choose_multiple(generator,number_of_centers).collect();
        
        let n_cords = test_data[0].0.nrows();

        let number_of_outputs = test_data[0].1.nrows();
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

            const MAX_TRAIN_ITEMS: usize = 30000;
            //dbg!(&epoch);
            test_data.shuffle(generator);
            let test_data = if test_data.len() > MAX_TRAIN_ITEMS {
                &test_data[..MAX_TRAIN_ITEMS]
            } else {
                &test_data
            };
            for (coords, results) in test_data {
                // inlined fn compute_outputs for access to hidden_outputs
                let (output, hidden_output) = self.compute_outputs_inner(coords.clone(), false);
            
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
                        // if delta.is_nan() {
                        //     let first = -1. * LR * (dbg!(output) - dbg!(result)) * dbg!(hidden_output);
                        //     dbg!(first);
                        //     dbg!(-1.  * dbg!(output * dbg!(1. - output)));
                        //     exit(1)
                        // }
                        if !delta.is_nan(){
                        weight.x += delta;
                        }
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
                    if !bias_delta.is_nan() {
                        bias.x += bias_delta;
                    }
                }
            }

            #[allow(unconditional_panic)] // rust will blame you for dividing by zero
            if max_epochs < NUM_INFO_PRINTS || // print every epoch
                (NUM_INFO_PRINTS != 0 && epoch % (max_epochs / NUM_INFO_PRINTS) == 0)
            {
                info!("Epoch: {}", epoch);
                let max_items  = test_data.len().min(MAX_STATS_ITEMS);
                self.print_statis(&test_data[..max_items]);

                let (accuracy, _error) = self.statis(&test_data[..max_items], false);
                if best_network.accuracy < accuracy {
                    info!("Replacing best network with network: seed={}, epoch={}, accuracy={}", seed, epoch, accuracy);
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
    }


    fn compute_outputs_inner(&self, coords: DVector<f64>, debug: bool) -> (DVector<f64>, DVector<f64>) {
        let n = self.centers.len();
        let hidden_output = DVector::from_fn(self.widths.nrows(), |row, _c| {
            let center_cord = &self.centers.column(row);
            let distance = (&coords - center_cord).norm();
            Self::rbf_gaussian(self.widths[row], distance)
        });
        if debug {
            dbg!(&self.weights);
            
            dbg!(&hidden_output);

            dbg!(&self.biases);
        }
        let output = (&self.weights * &hidden_output);
        let biased_output = output + &self.biases;

        let output = Self::soft_max(biased_output);
        (( output), hidden_output)
    }

    fn soft_max(coords: DVector<f64>) -> DVector<f64> {
       let results = coords.map(|c|c.exp());
       let sum = results.sum();
       results.map(|v|v/sum)

        // {
        //   // naive technique -- could easily over/under flow
        //   // see https://jamesmccaffrey.wordpress.com/2018/05/18/avoiding-an-exception-when-calculating-softmax/
        //   int n = vec.Length;
        //   double[] result = new double[n];
        //   double sum = 0.0;
        //   for (int i = 0; i < n; ++i)
        //     result[i] = Math.Exp(vec[i]);
        //   for (int i = 0; i < n; ++i)
        //     sum += result[i];
        //   for (int i = 0; i < n; ++i)
        //     result[i] /= sum;
        //   return result;
        // }
    }

    pub fn compute_outputs(&self, coords: DVector<f64>, debug: bool) -> DVector<f64> {
        self.compute_outputs_inner(coords,
        debug // debug
        ).0
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
    fn statis(&self, data: &[(DVector<f64>, DVector<f64>)], debug: bool) -> (f64, f64) {
        let mut sum_se = 0.;
        let mut nc = 0.;
        let mut nw = 0.;
        for (coords, results) in data {
            let outputs = self.compute_outputs(coords.clone(), debug);

            for (output, result) in outputs.row_iter()
                .zip(results.row_iter()) {
                let mut err = result.x - output.x;
                if debug {
                    dbg!(result.x);
                    dbg!(output.x);
                    dbg!(err);
                }
                if err > 1. {
                    err = 1.;
                } else if err < -1. {
                    err = -1.;
                } else if err.is_nan() {
                    err = 1.;
                }
                sum_se += err.powi(2);
            }
            let idx = Self::max_predict_index(outputs).unwrap();
            if results[idx] == 1. {
                nc += 1.;
            } else {
                nw += 1.;
            }
        }
        let accuracy = (nc * 1.0) / (nc + nw);
        let error = sum_se / data.len() as f64;
        (accuracy, error)
        
    }

    fn print_statis(&self, data: &[(DVector<f64>, DVector<f64>)]) {
        let (accuracy, error): (f64, f64) = self.statis(&data, false);
        info!("Error: {}", error);
        info!("Accuracy: {}", accuracy);
    }
    fn max_predict_index(result: DVector<f64>) -> Option<usize> {
        result.row_iter().enumerate().max_by(|(_, i1),(_, i2)|i1.x.partial_cmp(&i2.x).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0)
    }

    // Gaussian function that we used as radial basis in our neural network
    fn rbf_gaussian(width:f64, r: f64) -> f64 {
        (-(r / width).powi(2)).exp()
    }

    fn cmp_accuracy(&self, other: &Self, test_data: &[(DVector<f64>, DVector<f64>)]) -> std::cmp::Ordering {
        let our_stats = self.statis(test_data, false);
        let their_stats = other.statis(test_data, false);
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


fn init_test_data() -> Vec<(DVector<f64>, DVector<f64>)> {

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


    test_data_x.into_iter().zip(test_data_y.into_iter())
    .map(|(x,y)|
        
        (DVector::from_vec(x),

        DVector::from_iterator(3, y.into_iter().map(|i| i as f64)))

    ).collect()
}


fn example() {

    let test_data: Vec<(DVector<f64>, DVector<f64>)> = init_test_data();

    let max_epochs = 1000;
    let num_neurons = 15;
    execute(test_data, 0, 50, num_neurons, max_epochs);
    
    
}

fn execute(
    test_data: Vec<(DVector<f64>, DVector<f64>)>,
    start_seed: u64,
    num_networks: u64,
    hidden_neurons: usize,
    max_epochs: usize,
) {
    let mut best_network = None;
    for seed in start_seed..(start_seed + num_networks) {
        let data_elem = test_data.first().unwrap();
        let num_outputs = data_elem.1.nrows();
        let input_neurons = data_elem.0.nrows();
        
        // let seed = 11;
        let mut generator = rand::rngs::StdRng::seed_from_u64(seed);
        info!("Creating a {}-{}-{} RBF network, seed = {}", input_neurons, hidden_neurons, num_outputs,  seed);
        let sigma = 0.1;
        info!("Setting LR = {}, max_epochs = {}, sigma = {}", LR, max_epochs, sigma);
        info!("Starting training");
        let mut network = RBFNetwork::new( &mut generator, hidden_neurons, &test_data, sigma);
    
        let mut current_best = best_network.take().unwrap_or_else(||
        BestNetwork {
            seed,
            epoch: 0,
            accuracy: 0.,
            network: network.clone()
        });
        network.train(&mut generator, max_epochs, test_data.clone(), &mut current_best, seed);
        
        best_network = Some(current_best);
    }


    let best_network = best_network.unwrap();
    info!("The best network is on: seed={}, epoch={}", best_network.seed, best_network.epoch);

    let max_items  = test_data.len().min(MAX_STATS_ITEMS);
    best_network.network.print_statis(&test_data[..max_items]);

    info!("Statistic on full set:");
    best_network.network.print_statis(&test_data);
}


#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
enum Args {
    Example,
    Train {
        #[clap(long= "num-epochs")]
        /// How many iterations per networks should we process
        max_epochs: usize,
        #[clap(long= "num-networks")]
        /// How many networks to generate
        num_networks: u64,

        #[clap(long= "hidden-neurons")]
        hidden_neurons: usize,

        #[clap(long= "start-seed", default_value_t=0)]
        start_seed: u64,
        /// Some epochs iteration can increase network perfomance significant,
        /// some little increase perfomance, and other can reduce it.
        /// This flag give distribution of perfomance per epoch. 
        /// Check median epochs for maximum effort
        #[clap(long="epoch-stats")]
        epochs_stats: bool, // TODO


        // #[clap(long= "file")]
        // file: String,
    },
    Verify {
        #[clap(long= "num-epochs")]
        /// How many iterations per networks should we process
        max_epochs: usize,

        #[clap(long= "hidden-neurons")]
        hidden_neurons: usize,

        #[clap(long= "seed")]
        seed: u64,

        #[clap(long= "file")]
        file: String,
    },
    // /// Give information about file
    // Stats {

    //     #[clap(long= "file")]
    //     file: String,
    // },

    // Take {

    //     #[clap(long= "file")]
    //     file: String,
    // }

}

struct KDDSchemaReader {
}

impl KDDSchemaReader {
    pub fn parse_line(line: &str) -> Result<KDDRecord, anyhow::Error> {
        let lines: Vec<_> = line.split(',').collect();
        let record_raw = csv::ByteRecord::from(lines);
        record_raw.deserialize(
            None, // Names in KDDRecord are same
        ).map_err(From::from)
    }
    pub fn parse_line_labeled(line: &str) -> Result<KDDRecordWithLable, anyhow::Error> {
        let line = line.trim_matches('.');
        let lines: Vec<_> = line.split(',').collect();
        let record_raw = csv::ByteRecord::from(lines);
        record_raw.deserialize(
            None, // Names in KDDRecord are same
        ).map_err(From::from)
    }

    pub fn test_data(file: &Path) -> Result<(Vec<(DVector<f64>, DVector<f64>)>, Vec<KDDRecordWithLable>), anyhow::Error> {

        debug!("Reading file {:?}", file);
        let kdd_file = File::open(&file)?;
        let decoder = BufReader::new(GzDecoder::new(kdd_file));
               
        let mut test_data = vec![];
        for line in decoder.lines() {  
            let name = line?;
            debug!("string = {:?}", name);
            let record = KDDSchemaReader::parse_line_labeled(&name)?;
            debug!("record = {:?}", record);
            test_data.push(record)
        }

        info!("Total records loaded: count={}", test_data.len());
        
        let mut test_data_vectored:Vec<_> = test_data.iter().map(|r|
            (r.input_vector(), r.output_vector())).collect();
        Ok((test_data_vectored, test_data))
    }
}
fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    env_logger::init();
    
    match args {
        Args::Example => example(),
        Args::Train {
            max_epochs,
            num_networks,
            start_seed,
            hidden_neurons,
            epochs_stats: _epochs_stats
        } => {
            const KDD_TRAIN_DATA: &str = "data/kddcup.data_10_percent.gz";

            let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
            let kdd_data_path = PathBuf::from(project_dir).join(KDD_TRAIN_DATA);

            let test_data_vectored = KDDSchemaReader::test_data(&kdd_data_path)?;
            execute(test_data_vectored.0,
            start_seed,
            num_networks,
            hidden_neurons,
            max_epochs)
        }
        Args::Verify { 
            seed,
            max_epochs,
            hidden_neurons,
            file
        } => {
            const KDD_TRAIN_DATA: &str = "data/kddcup.data_10_percent.gz";

            let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
            let kdd_data_path = PathBuf::from(project_dir).join(KDD_TRAIN_DATA);

            let (test_data,_) = KDDSchemaReader::test_data(&kdd_data_path)?;

            // Recover network 

            let data_elem = test_data.first().unwrap();
            let num_outputs = data_elem.1.nrows();
            let input_neurons = data_elem.0.nrows();
            
            // let seed = 11;
            let mut generator = rand::rngs::StdRng::seed_from_u64(seed);
            info!("Creating a {}-{}-{} RBF network, seed = {}", input_neurons, hidden_neurons, num_outputs,  seed);
            let sigma = 0.1;
            info!("Setting LR = {}, max_epochs = {}, sigma = {}", LR, max_epochs, sigma);
            info!("Starting training");
            let mut network = RBFNetwork::new( &mut generator, hidden_neurons, &test_data, sigma);
        
            let mut current_best = 
            BestNetwork {
                seed,
                epoch: 0,
                accuracy: 0.,
                network: network.clone()
            };
            network.train(&mut generator, max_epochs, test_data.clone(), &mut current_best, seed);
        
           
            let kdd_data_path = PathBuf::from(file);

            let verify_data = KDDSchemaReader::test_data(&kdd_data_path)?;

            info!("Statistic on full set:");
            print_statis_custom(&network, &verify_data.0);
        }
        _ => todo!()
    }
    Ok(())



}


fn statis_custom(network: &RBFNetwork, data: &[(DVector<f64>, DVector<f64>)], debug: bool) -> (f64, f64, f64, f64, f64) {
    let mut sum_se = 0.;
    let mut nc = 0.;
    let mut nw = 0.;

    let  (mut not_ddos_catches, mut not_ddos_total) = (0., 0.);

    let  (mut ddos_catches, mut ddos_total) = (0., 0.);

    let  (mut ddos_in_data, mut ddos_miss_in_data) = (0., 0.);
    for (coords, results) in data {
        let outputs = network.compute_outputs(coords.clone(), debug);

        for (output, result) in outputs.row_iter()
            .zip(results.row_iter()) {
            let mut err = result.x - output.x;
            if debug {
                dbg!(result.x);
                dbg!(output.x);
                dbg!(err);
            }
            if err > 1. {
                err = 1.;
            } else if err < -1. {
                err = -1.;
            } else if err.is_nan() {
                err = 1.;
            }
            sum_se += err.powi(2);
        }
        let idx = RBFNetwork::max_predict_index(outputs).unwrap();
        if results[idx] == 1. {
            nc += 1.;
        } else {
            nw += 1.;
        }

        // count ddos statistic based on last index
        let not_ddos_idx = results.nrows() - 1;
        if results[not_ddos_idx] == 1. {
            not_ddos_total+= 1.;
            if idx == not_ddos_idx {
                not_ddos_catches += 1.;
            }
        } else {
            ddos_total  += 1.;
            if idx != not_ddos_idx {
                ddos_catches += 1.;
            }
        }

        if results[not_ddos_idx] == 1. {
            ddos_miss_in_data += 1.;
        } else {
            ddos_in_data += 1.;
        }

    }
    let accuracy = (nc * 1.0) / (nc + nw);
    let ddos_accuracy = (ddos_catches * 1.0) / (ddos_total);

    let not_ddos_accuracy = (not_ddos_catches * 1.0) / (not_ddos_total);

    let ddos_in_data_rate = (ddos_in_data * 1.0) / (ddos_in_data + ddos_miss_in_data);
    let error = sum_se / data.len() as f64;
    (accuracy, error, ddos_accuracy, not_ddos_accuracy, ddos_in_data_rate)
    
}

fn print_statis_custom(network: &RBFNetwork, data: &[(DVector<f64>, DVector<f64>)]) {
    let (accuracy, error, ddos_accuracy, not_ddos_accuracy, ddos_in_data_rate) = statis_custom(network, &data, false);
    info!("Error: {}", error);
    info!("Accuracy: {}", accuracy);
    info!("Data statistic: {}", ddos_in_data_rate);
    info!("False negative: {}", 1. - not_ddos_accuracy);
    info!("False positive: {}", 1. - ddos_accuracy);
}