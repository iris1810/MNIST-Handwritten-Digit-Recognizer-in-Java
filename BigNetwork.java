///////////////////////////////////////
// Name: Khai Tran Nguyen
// Class: CSC 475
// Date : Oct 21 2025
// Part 2: MNIST Handwritten Digit Recognizer
///////////////////////////////////////

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Random;

/**
 * BigNetwork class implements a simple feedforward neural network with one hidden layer.
 */
public class BigNetwork {
    
    //for the varieable: 
    // we need shape W1[15][784], b1[15]; W2[10][15], b2[10]
    double [][] W1, W2;
    double [] b1, b2;

    final double learning_rate; // suggested 3.0
    final int batch_size; // minibatch size of 10,and 30 epochs


    // ---------- Caches ----------// what we do here like do the structure 
    static final class Forward { // a nested class iniside BigNetwork
        double[] x;         // a^0
        double[] z1, a1;    // layer 1
        double[] z2, a2;    // layer 2 (output)
    }

    // save the object for a single gradient decent 
    // [current_layer][previous_layer]
    static final class Gradients {
        double [][] dW1 = new double [15][784]; // layer 0->1
        double [] db1 = new double [15]; // layer 0->1
        double [][] dW2 = new double [10][15]; // layer 1->2
        double [] db2 = new double [10]; // layer 1->2   
        
        // elementwise add into this (for minibatch accumulation)
            // for w1, b1
        void add_grad( Gradients g){  // g come from one training sample per one minibatch.
            for(int i=0; i< dW1.length; i++){ 
                for (int j=0; j< dW1[0].length; j++){
                    dW1[i][j] += g.dW1[i][j];  // for the same [i][j], from the object g, get that [i][j]
                                            // add to the memory of dW1[i][j] 
                                            // => it the summation of each sample in a minibatch whith the same node [i][j]
                                            // it start from 0, then add in each sample in the minibatch
                }
            }
            for(int i=0; i< db1.length; i++){
                    db1[i] += g.db1[i]; // for each [i] node, from object g,  get that then add in, 
                                    // so that we can get the summation when we done the minibatch
            }

            //for W2,b2
            for(int i=0; i< dW2.length; i++){
                for (int j=0; j< dW2[0].length; j++){
                    dW2[i][j] += g.dW2[i][j];
                }
            }
            for(int i=0; i< db2.length; i++){
                db2[i] += g.db2[i];
            }
        }


        void zero_grad (){
            // set all to 0 after each minibatch
            for(int i=0 ; i < dW1.length; i++){
                for (int j = 0; j < dW1[0].length; j++) {
                    dW1[i][j] = 0.0;
                }
            }

            for(int i=0; i< db1.length; i++){
                db1[i] = 0.0;
            }

            for(int i=0 ; i < dW2.length; i++){
                for (int j = 0; j < dW2[0].length; j++) {
                    dW2[i][j] = 0.0;
                }
            }

            for(int i=0; i< db2.length; i++){
                db2[i] = 0.0;
            }
        }
    }

    /////constructor /////
    public BigNetwork(double learning_rate, int batch_size) { // -> it will collect the data 
        this.learning_rate = learning_rate;
        this.batch_size = batch_size;
        initRandom();
    }

    ////// Initialization //////
    private void initRandom(){
        Random random_seed = new Random(42); // fixed seed for reproducibility

        // allocate
        W1 = new double[15][784];
        b1 = new double[15];
        W2 = new double[10][15];
        b2 = new double[10];

        for (int i=0; i<15; i++){
            for (int j=0; j<784;j++){
                // create random W1 between -1 and 1
                W1[i][j] = -1.0 + 2.0 * random_seed.nextDouble(); // [-1,1] 
                // nextDouble is a value in between [0, 1).
                // multiple by 2 => [0,2), then -1 => [-1,1)
            }
            // random b1 between -1 and 1
            b1[i] = -1.0 +2.0 * random_seed.nextDouble();
        }

        for (int i=0; i<10; i++){
            for (int j=0; j<15;j++){
                // create random W2 between -1 and 1
                W2[i][j] = -1.0 + 2.0 * random_seed.nextDouble(); // [-1,1] 
                // nextDouble is a value in between [0, 1).
                // multiple by 2 => [0,2), then -1 => [-1,1)
            }
            // random b2 between -1 and 1
            b2[i] = -1.0 +2.0 * random_seed.nextDouble();
        }
    }

    ////// Forward pass //////
    public Forward forward (double [] x){
        Forward f = new Forward();
        f.x = x;
        // z^1 = W1 a^0 + b1
        f.z1 = Matrix.add( Matrix.matrix_vec_mul(W1,x), b1);
        // a^1 = σ(z^1)
        f.a1 = Matrix.sigmoid(f.z1);

        // z^2 = W2 a^1 + b2
        f.z2 = Matrix.add(Matrix.matrix_vec_mul(W2,f.a1),b2);
        // a^2 = σ(z^2)
        f.a2 = Matrix.sigmoid(f.z2);

        return f;
    }
    // backpropagation for one (x,y) 
    public Gradients backprop (double [] x, double[]y ){
        Forward f = forward(x);
        Gradients g = new Gradients ();

        // δ^L = (a^L - y) ⊙ σ'(z^L)  (use σ'(z) = a ⊙ (1-a))
        double [] right2 = Matrix.sigmoidPrimeFromA(f.a2);
        double [] left2 = new double [10]; // 10 nodes in output layer
        for (int i=0; i < 10; i++){
            left2[i] = f.a2[i] - y[i]; 
        }
        double [] delta2 = Matrix.hadamard(left2,right2);

        // ∇b^L = δ^L,  ∇W^L = δ^L (a^{L-1})^T
        g.db2 = delta2.clone(); // clone to make a copy of the array
        g.dW2 = Matrix.outer(delta2, f.a1); // it still the multiple, 
                                     // delta2 is [10], a1 is [15], outer produce [10][15] matrix
        // δ^1 = (W^2)^T δ^2 ⊙ σ'(z^1)
        double [] left1 =  new double [15]; // 15 nodes in layer 1
        for (int j=0; j< 15; j++){
            double s = 0.0;
            for (int k=0 ; k<10; k++){
                s += W2[k][j] * delta2[k]; // W2^T * delta2
            }
            left1[j] = s;
        }
        double [] right1 = Matrix.sigmoidPrimeFromA(f.a1);
        double [] delta1 = Matrix.hadamard(left1, right1);

        // ∇b^1 = δ^1,  ∇W^1 = δ^1 (a^0)^T
        g.db1 = delta1.clone();
        g.dW1 = Matrix.outer(delta1,f.x);

        return g;
    }


    //=== Apply one minibatch gradient (average then descent) ===
    public void applyBatch(Gradients sum, int m){
        double s = learning_rate/m; // η/m
        // for layer 2;
        //W2 := W2 - s * dW2 and  b2 := b2 - s * db2 
        for (int i=0; i < W2.length; i++){
            for (int j=0; j< W2[0].length;j++){
                W2[i][j] -= s * sum.dW2[i][j]; // elementwise updates
            }
            // update b2
            b2[i] -= s * sum.db2[i];
        }
        //similarly for layer 1
        for(int i=0; i< W1.length; i++){
            for (int j=0; j < W1[0].length;j++){
                W1[i][j] -= s * sum.dW1[i][j]; // elementwise updates
            }
            // update b1
            b1[i] -= s * sum.db1[i];
        }
    }

    // This class is a helper class
    // contain all the matrix operation that we need
    // for the neural network forward and backpropagation
    public static class  Matrix {
        private Matrix() {}  // static methods only 

        // y = W · x  (W: r×c, x: c) normal nultiple
        public static double [] matrix_vec_mul (double[][] w, double[] x){
            int r = w.length;        // number of rows
            int c = w[0].length;     // number of columns
            double [] y = new double [r];
            for (int i=0; i<r; i++){
                double s = 0.0;
                for (int j=0; j<c; j++){
                    s += w[i][j] * x[j];
                }
                y[i]= s;
            }
            return y;
        }

        // z = u ⊙ v  (Hadamard  product)
        public static double [] hadamard(double[] u, double[] v){
            double[] z = new double[u.length];
            for (int i = 0; i < u.length; i++) {
                z[i] = u[i] * v[i];
            }
            return z;
        }
        // adding 2 vector together, return a new array
        public static double [] add (double[] u, double[] v){
            double [] result = new double [u.length];
            for (int i=0; i< u.length; i++){
                result[i] = u[i] + v[i];
            }
            return result;
        }
        
        //adding 2 vectors but replacement with the old one
        public static void addInPlace(double[] a, double[] b) {
            for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
            }
        }

        // Outer product: δ (r) × a^T (c) → r×c . [3x1][4x1]=[3x4] for backprop
        public static double [][] outer (double[] delta, double[] a){
            int r = delta.length;
            int c = a.length;
            double [][] result = new double [r][c];
            for (int i=0; i<r; i++){
                for (int j=0; j< c; j++){
                    result[i][j] = delta[i] * a[j];
                }
            }
            return result;
        }   

        // σ(z) and σ'(z) elementwise
        public static double[] sigmoid (double[] z){
            double [] result = new double [z.length];
            for (int i=0; i< z.length; i++){ 
                result[i] = 1.0 / (1.0 + Math.exp(-z[i])); // Math.exp is e^x  
            }
            return result;
        }
        public static double [] sigmoidPrimeFromA (double [] a){
            double [] result = new double[a.length];
            for (int i=0; i< a.length; i++){
                result[i] = a[i] * (1.0 - a[i]);
            }
            return result;
        }

        // utilities
        // pretty printers
        public static String fmt(double[] v){
            StringBuilder stringb = new StringBuilder();
            stringb.append('[');
            for (int i=0; i< v.length; i++){
                if ( i>0){
                    stringb.append(",");
                }
                stringb.append(String.format("%.6f",v[i]));
            }
            stringb.append ("]");
            return stringb.toString();
        }   
    }
    ///////////// Adding more for part 2 /////////////

    // Using for loading the MNist CSV file
    public static class MnistIO {
        public static class Record{
            double [] x; // input
            double [] y; // output
            int label; // use this to create the output, 
                        // one hot encoded label (10 classes)
        
            public Record(double []x, int label){
                this.x = x;
                this.label= label;
                this.y = oneHot(label); // “One-hot encoding” 
                                //is a way to represent a class label as a vector (number into vector)
            }
        }
        // one hot encoding for label - a way to represent a class label as a vector
        private static double [] oneHot (int label){
            // we have 10 digits class; then each index of vector represent each digit
            double[] vec = new double [10];
            vec[label] = 1.0; // set the index of the label to 1, the rest still keep 0
            return vec;
        }

        // load the MNist CSV file , store list of element type Record
        public static List<Record> loadCSV (String filename){
            List <Record> data = new ArrayList<>();
            // Every time you read one line from the CSV, you create one Record and add it to data.
            // dynamic array
            File file = new File(filename);

            try(Scanner reader = new Scanner(file)){
                while(reader.hasNextLine()){
                    String line = reader.nextLine();
                    String[] tokens = line.split(",");

                    // first collumn = label
                    int label = Integer.parseInt(tokens[0]);

                    // Next 784 collums = pixel values
                    double [] x = new double[784];
                    for (int i=0; i<784; i++){
                        // Fill the input array x[] for one image
                        x[i] = Integer.parseInt(tokens[i+1])/255.0; // normalize pixel value to [0,1]
                        //because Each pixel is a grayscale value between: 0 (black) and 255 (white)
                        // values in between are different shades of gray.
                    }
                    data.add(new Record(x,label));
                }
            } catch (FileNotFoundException e){
                System.out.printf("File not found: %s%n", filename);
                e.printStackTrace();
            }
            return data; // return the list of Record
        }
    }
    // Add evaluation (per-digit + overall) and argmax
    static int argMax (double [] a){
        int maxIndex =0;
        for (int i=0; i<a.length;i++){
            if (a[i] > a[maxIndex]){
                maxIndex =i;
            }
        }return maxIndex;
    }

    // A custom class to store and organize accuracy statistics per digit.
    static final class Statistic{
        int[] correct = new int[10];
        int[] total = new int[10];
    }

    public Statistic evaluate(List<MnistIO.Record> data){
        Statistic s = new Statistic ();
        for (MnistIO.Record record : data){
            // for each record, do forward pass to get the output a2
            // for a2, find the argamax to get the predicted digit
            int predict = argMax(forward(record.x).a2);
            // for the real number (label), increment the index of total in that label index
            // if the true label is 3, then we increment s.total[3] by 1.
            // (increasing 1 more times this number occur)
            s.total[record.label]++;

            // then compare does the predit is correct with that true label
            if (predict == record.label){
                s.correct[record.label]++; // if correct, increment the correct count for that label
            }
        }
        return s;
    }

    // print the evaluation result
    static void printEpochReport(int epoch, Statistic s){
        int correctP = 0;
        int totalP =0;
        System.out.printf("===== Epoch %d =====%n", epoch+1);
        for (int digit =0; digit <10; digit ++){
            correctP += s.correct[digit];
            totalP += s.total[digit];
            System.out.printf("Digit %d: %d/%d%n", digit, s.correct[digit],s.total[digit]);
        }
        double acc;
        if (totalP ==0){
            acc = 0.0;
        }else{
            acc = 100.0 * correctP / totalP;
        }
        System.out.printf("Accuracy: %d/%d = %.2f%%%n", correctP, totalP, acc);
        // %% to print a single % sign
    }
    // we shuffle the training data (Fisher–Yates)
    // for every epoch, we want it see the training data with different order
    // so it can redice the overfitting when it just keep the same order
    static void shuffleIndices(int[] index, Random random){
        // Fisher–Yates shuffle
        for(int i= index.length -1; i>0; i--){
            // pick a random index from 0 to i
            int j= random.nextInt(i+1);
            // swap index[i] with index[j]
            int temp = index[i];
            index[i] = index[j];
            index[j] = temp;    
        }
    }

    //// Train for 1 epoch ////     
    void trainOneEpoch (List<MnistIO.Record> training){
        int n = training.size();

        // build an index array 0..n-1. 
        // so you can get a new random order without changing the dataset.
        // ex: idx = [3, 1, 4, 0, 2]
        int [] index = new int[n];
        for(int i=0; i<n; i++){
            index[i] = i;
        }
        //shuffle the indices so we visit samples in random order
        Random rnd = new Random();              // or new Random(42) for reproducible runs
        shuffleIndices(index, rnd);

        // for each minibatch, use the shuffled order
        for (int start =0; start <n ; start+=batch_size){ // batch size is the amount of each minibatch
            int end = Math.min(start+ batch_size, n); // make sure we don't go out of bound
            Gradients sum = new Gradients();
            sum.zero_grad(); // set all to 0 before each minibatch

            //Accumulate gradients over the current minibatch
            for(int t = start; t< end; t++){
                MnistIO.Record r = training.get(index[t]); // get the record at the shuffled index)
                            // might be the shuffle list is idx = [3, 1, 4, 0, 2]
                            // then after we get(3), get(1), get(4)... get the record in random order not 
                            // not idx = [0,1,2,3,4] get(0), get(1), get(2)
                Gradients g = backprop(r.x, r.y);
                sum.add_grad(g); // accumulate the gradient
            }
            int m = end - start;                // actual batch size for this final chunk
                            // Because the last batch may not be full.
            applyBatch(sum, m);
        }
    }

    /////////// ASCII render /////////////////
    // 1. These are the ASCII characters used to "draw" the image.
        //  from light (space ' ') to dark ('@').
    static final char[] ASCII_RAMP = " .:-=+*#%@".toCharArray();
        // index  0 1 2 3 4 5 6 7 8 9  (10 total levels of darkness)
        // space .  :   -   =   +   *   #   %   @.  (light -> dark) (black -> white)
        // grayscale was 0-255, but during the csv loading, we already normalize it to 0.0-1.0 (black - white)

    // 2. convert a 28x28 MNist grayscale image (0->1) into ASCII art 
    static String toAscii (double[] x){
        //stringBuilder to store the ASCII character
        // 28x28 need 28 pixel per row + 1 newline character (/n) per row 
        // 28* (28+1)
        StringBuilder art_build = new StringBuilder (28*(28+1));

        // loop over each pixel row 
        for (int r=0; r<28; r++){
            for (int c=0; c<28; c++){

                // 1.get the pixel value
                // Each MNIST image is an array of length 784 (28x28)
                // r*28+c gives the index for pixel (r, c)
                double get_pixel = x[r * 28 + c]; 

                // 2. map the pixel to ascii index
                // Multiply by (ramp - 1) to map to range [0..9] 
                int map = (int) Math.round(get_pixel * (ASCII_RAMP.length -1));

                // make sure in range 0-9
                if (map <0) map =0;
                if (map >= ASCII_RAMP.length) map = ASCII_RAMP.length -1 ;

                // 4. append the ASCII to the picture 
                art_build.append(ASCII_RAMP[map]);
            }
            // 5. end the row -> enter to new line
            art_build.append("/n");
        }
        return art_build.toString();
    }

    /////// Test viewer //////////
    // Option 5 show every testing image (press 1 to continue) 
    static void browseTestingImage (BigNetwork net, List<MnistIO.Record> test){
        // read keyboard input from the console
        Scanner in = new Scanner(System.in);

        // Loop through each test sample
        for (int i =0; i < test.size(); i++){
            // get the ith record from record for testing
            MnistIO.Record current_record = test.get(i);

            // do forward with x, argmax to guess the digit from a2, check the correction
            int predict = argMax(net.forward(current_record.x).a2);
            boolean correct = (predict == current_record.label);

            // display the information
            // - network's predicted output
            // - status (Correct / Incorrect)
            System.out.printf("Testing Case #%d.  Correct classification = %d.   Network Output = %d.  %s%n", i, current_record.label, predict, correct ? "Correct." : "Incorrect.");

            // print picture testing
            System.out.println(toAscii(current_record.x)); // x just the grayscale value

            // Ask the user whether to continue or return to the menu
            System.out.print("Enter 1 to continue. All other values return to main menu: ");

            // read input
            String read = in.nextLine().trim();
            // if not 1, break
            if (!read.equals("1")) break ;
        }
    }
    // option 6 Show only misclassified testing images
    //  Displays only the ones where the network predicted incorrectly
    static void browseMisclassified(BigNetwork net, List<MnistIO.Record> test){
        // read keyboard input from the console
        Scanner in = new Scanner(System.in);

        // keep track how many misclassified images we showed
        int shown =0;

        // Loop through each test sample
        for (int i =0; i < test.size(); i++){
            // get the ith record from record for testing
            MnistIO.Record current_record = test.get(i);

            // do forward with x, argmax to guess the digit from a2, if it corrrect, skip to the next image 
            int predict = argMax(net.forward(current_record.x).a2);
            if (predict == current_record.label) continue;

            // display if the network made a mistake
            // - network's predicted output
            // - status (Correct / Incorrect)
            System.out.printf("Testing Case #%d.  Correct classification = %d.  Network Output = %d.  Incorrect.%n", i, current_record.label, predict);

            // print picture testing
            System.out.println(toAscii(current_record.x)); // x just the grayscale value

            // Ask the user whether to continue or return to the menu
            System.out.print("Enter 1 to continue. All other values return to main menu: ");

            // read input
            String read = in.nextLine().trim();
            // if not 1, break
            if (!read.equals("1")) break ;

            // increase the amount missclassified images
            shown++;
        }
        
        // If no misclassified images were found, print a friendly message
        if (shown == 0) System.out.println("No misclassified images. Good job!");
    }

    // option 7 SAVE the trained neural network weights & biases
    // into text saving file
    void saveText(String path){
        try{
            // FileWriter: writes characters to a file
            FileWriter file_w = new FileWriter(path);
            // makes it easy to print formatted text
            PrintWriter print_w = new PrintWriter(file_w); 

            // Save W1 (shape 15 x 784) r x c, each row have 784 value
            // write the matrix dimensions, use to load in later
            print_w.println(15 + " " + 784);
            for (int i= 0; i <15; i++){
                for (int j=0; j<784; j++){
                    //print to file W1 + " "
                    print_w.print(W1[i][j] + " ");
                }
                // move to the next row
                print_w.println();
            }

            // Save b1 (shape 15)
            print_w.println(15);
            for (int i=0 ;i<15;i++){
                print_w.print(b1[i] + " ");
            }
            print_w.println(); // new line at the end of b1

            // Save W2 (shape 10 x 15) r x c, each row have 784 value
            // write the matrix dimensions, use to load in later
            print_w.println(10 + " " + 15);
            for (int i= 0; i <10; i++){
                for (int j=0; j<15; j++){
                    //print to file W1 + " "
                    print_w.print(W2[i][j] + " ");
                }
                // move to the next row
                print_w.println();
            }

            // Save b2 (shape 10)
            print_w.println(10);
            for (int i=0;i<10;i++){
                print_w.print(b2[i] + " ");
            }
            print_w.println(); // new line at the end of b1

            //// Close the stream 
            print_w.close();
            file_w.close();

            System.out.println("Save network to: " + path);
        } catch (IOException e){
            //print error
            System.out.println("Save failed " + e.getMessage());
        }
    }

    ///// Option 2 Load Network Weight and bias from a Text file 
    void loadText(String path){
        try{
            //read numbers
            Scanner scan = new Scanner(new File(path));

            //load W1 15x784 
            int r1 = scan.nextInt(); // rows = 15
            int c1 = scan.nextInt(); // cols = 784
            W1 = new double[r1][c1];
            for (int i=0; i<r1; i++){
                for(int j=0; j<c1;j++){
                    // read value
                    W1[i][j] = scan.nextDouble();
                }
            }

            //Load b1
            int b1_scan = scan.nextInt(); // b1 =15
            b1 = new double[b1_scan];
            for (int i=0; i<b1_scan; i++){
                b1[i] = scan.nextDouble();
            }

            //load W2 10x15
            int r2 = scan.nextInt(); // rows = 10
            int c2 = scan.nextInt(); // cols = 15
            W2 = new double[r2][c2];
            for (int i=0; i<r2; i++){
                for(int j=0; j<c2;j++){
                    // read value
                    W2[i][j] = scan.nextDouble();
                }
            }

            //Load b2
            int b2_scan = scan.nextInt(); // b1 =10
            b2 = new double[b2_scan];
            for (int i=0; i<b2_scan; i++){
                b2[i] = scan.nextDouble();
            }

            //close
            scan.close();
            System.out.println("Loaded network from: " + path);
        } catch (IOException e){
            System.out.print("Load failed" + e.getMessage());
        }
    }


    // Option 3,4 show accuracy Train or Test data
    static void showTrainAccuracy(BigNetwork net, List<MnistIO.Record> train){
        Statistic s = net.evaluate(train);
        printEpochReport(0, s);
    }
    static void showTestAccuracy(BigNetwork net, List<MnistIO.Record> test){
        Statistic s = net.evaluate(test);
        printEpochReport(0, s);
    }

    //// MAIN /////
    public static void main (String[] args){
        // 1.Load CSV data
        List<MnistIO.Record> train = MnistIO.loadCSV("mnist_train.csv");
        List<MnistIO.Record> test = MnistIO.loadCSV("mnist_test.csv");
        System.out.println("Loaded training samples:" + train.size());
        System.out.println("Loaded testing samples:" + test.size());

        // 2.Create the BigNetwork
        BigNetwork net = new BigNetwork(3.0, 10); // learning rate = 3.0, minibatch size = 10
        boolean ready = false;  

        // scanner to read user input
        Scanner user_input = new Scanner(System.in);

        ///// 3. CLI Main Menu loop /////
        while (true){
            System.out.println("\n=== Main Menu ===");
            System.out.println("1. Train the network");
            System.out.println("2. Load a pre-trained network");
            System.out.println("3. Display network accuracy on training data");
            System.out.println("4. Display network accuracy on testing data");
            System.out.println("5. Run network on testing data showing images and labels");
            System.out.println("6. Display the misclassified testing images");
            System.out.println("7. Save the network state to file");
            System.out.println("0. Exit");
            System.out.print("Select: ");

            // Read user input
            String user_choice = user_input.nextLine().trim();

            // Handle user option
            switch (user_choice){
                // option 1 : train the network
                case "1":{
                    // enter E for choose epoch, T for until reach the target
                    System.out.print("Train by (E)pochs or (T)arget accuracy (ex: 0.99)? ");
                    String mode = user_input.nextLine().trim().toUpperCase(); // make sure it uppercase

                    // "T" target mode
                    if (mode.startsWith("T")){
                        System.out.print("Your Target accuracy between 0.0 - 1.0 is :  ");
                        double target = Double.parseDouble(user_input.nextLine().trim());
                        int maxEpochs = 45; // limit so training doesn't go forever

                        for (int e=0; e< maxEpochs; e++){
                            net.trainOneEpoch(train);
                            Statistic s = net.evaluate(test);
                            printEpochReport(e, s);

                            // stop near 99%
                            int correct =0;
                            int total =0;

                            for (int d=0; d<10; d++){
                                correct += s.correct[d];
                                total += s.total [d];
                            }
                            //1.0 * 95 / 100 = 0.95 , need 0.99, mean that 99/100 correct
                            if (total > 0 && (1.0*correct/total) >= target) break;
                        }

                    } // "E" Epochs mode
                    else {
                        System.out.print("Epochs to train (max is 45): ");
                        int epochs = Integer.parseInt(user_input.nextLine().trim());

                        for (int e=0; e < epochs; e++){
                            net.trainOneEpoch(train);
                            Statistic s = net.evaluate(test);
                            printEpochReport(e, s);
                        }
                    } 
                    // Accuracy prompt after training
                    ready = true;
                    break;
                }
                // case 2 Load a pre-trained model
                case "2":{
                    System.out.print("Path to load: ");
                    String path = user_input.nextLine().trim();
                    net.loadText(path);
                    // Accuracy prompt after training
                    ready = true;
                    break;
                }

                // case 3 Show accuracy on training data
                case "3":{
                    if (!ready) { System.out.println("Please train (1) or load (2) first."); break; }
                    showTrainAccuracy(net, train);
                    break;
                }

                // case 4 Show accuracy on testing data
                case "4":{
                    if (!ready) { System.out.println("Please train (1) or load (2) first."); break; }
                    showTestAccuracy(net, test);
                    break;
                }

                //case 5 Browse testing images one by one (ASCII view)
                 case "5":{
                    if (!ready) { System.out.println("Please train (1) or load (2) first."); break; }
                    browseTestingImage(net, test);
                    break;
                }

                // case 6 Show only misclassified testing images
                 case "6":{
                    if (!ready) { System.out.println("Please train (1) or load (2) first."); break; }
                    browseMisclassified(net, test);
                    break;

                }
                // case 7 Save current model
                 case "7":{
                    if (!ready) { System.out.println("Please train (1) or load (2) first."); break; }
                    System.out.println("Path to save: ex: model.txt");
                    String path = user_input.nextLine().trim();
                    net.saveText(path);
                    break;
                }

                // case 0 Exit the program
                case "0":{
                    System.out.println("Bye!");
                    // Important: close Scanner before exiting
                    user_input.close();
                    return;
                }

                // default
                default: 
                    System.out.println("Invalid choice.");

            }
        }
    }
 }




