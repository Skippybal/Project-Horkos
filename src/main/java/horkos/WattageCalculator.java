package horkos;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;

import java.io.File;
import java.io.IOException;

public class WattageCalculator {

    public static void main(String[] args) throws IOException {
        loadData();
    }

    private static void loadData() throws IOException {
        try(RecordReader recordReader = new CSVRecordReader(1, ',')){
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("data/nucDataset.csv").getFile()
            ));

            //8983

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 200, 4, 4, true);
            DataSet allData = iterator.next();
            allData.shuffle(666);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            NormalizerSerializer saver = NormalizerSerializer.getDefault();
            File normalsFile = new File("models/Normalizers/nucNormalizer.zip");
            saver.write(normalizer,normalsFile);


            //System.out.println(allData);
            //iterator.setPreProcessor(normalizer);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.66);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTrain();

            NucModel.nucNetwork(trainingData, testData, "models/nucModel.zip", 0.001);



        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        try(RecordReader recordReader = new CSVRecordReader(1, ',')){
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("data/binDataset.csv").getFile()
            ));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 200, 4, 4, true);
            DataSet allData = iterator.next();
            allData.shuffle(666);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            NormalizerSerializer saver = NormalizerSerializer.getDefault();
            File normalsFile = new File("models/Normalizers/binNormalizer.zip");
            saver.write(normalizer,normalsFile);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.66);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTrain();

            //NucModel.nucNetwork(trainingData, testData, "models/binModel.zip", 0.001);



        } catch (InterruptedException e) {
            e.printStackTrace();
        }


    }

}
