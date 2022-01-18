package horkos;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

public class NucModel {
    private static final int feature_count = 4;

    public static void nucNetwork(DataSet train, DataSet test, String saveFile, double learningRate){
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                //.updater(new Nesterovs(0.1,0.9))
                //.updater(new Adam(0.0015))
                .updater(new Adam(learningRate))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(feature_count).nOut(16).build())
                .layer(1, new DenseLayer.Builder().nIn(16).nOut(16).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .activation(Activation.IDENTITY).nIn(16).nOut(1).build())
//                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
//                        .activation(Activation.IDENTITY).nIn(32).nOut(1).build())
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        model.setListeners(new StatsListener(statsStorage));


        //model.setListeners(new ScoreIterationListener(20));

        int nEpochs = 8000;

        for (int i = 0; i < nEpochs; i++) {
            model.fit(train);
        }

        //System.out.println(config.toJson());

        //model.fit(train);

        INDArray output = model.output(test.getFeatures());
        RegressionEvaluation eval = new RegressionEvaluation();

//        Iterator i =  test.iterator();
//        while (i.hasNext()){
//            DataSet t = (DataSet) i.next();
//            INDArray out = t.getFeatures();
//            INDArray labels = t.getLabels();
//            INDArray predicted = model.output(out, false);
//            eval.eval(labels, predicted);
//        }
//        System.out.println(eval.stats());

        // http://localhost:9000/train/overview

        eval.eval(test.getLabels(), output);
        System.out.printf(eval.stats());

        try {
            model.save(new File(saveFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
