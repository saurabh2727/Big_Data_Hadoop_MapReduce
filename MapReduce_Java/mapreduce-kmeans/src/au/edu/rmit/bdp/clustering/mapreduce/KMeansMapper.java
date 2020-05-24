package au.edu.rmit.bdp.clustering.mapreduce;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import au.edu.rmit.bdp.distance.DistanceMeasurer;
import au.edu.rmit.bdp.distance.EuclidianDistance;
import au.edu.rmit.bdp.clustering.model.Centroid;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;

import au.edu.rmit.bdp.clustering.model.DataPoint;

/**
 * First generic specifies the type of input Key.
 * Second generic specifies the type of input Value.
 * Third generic specifies the type of output Key.
 * Last generic specifies the type of output Value.
 * In this case, the input key-value pair has the same type with the output one.
 *
 * The difference is that the association between a centroid and a data-point may change.
 * This is because the centroids has been recomputed in previous reduce().
 */
public class KMeansMapper extends Mapper<Centroid, DataPoint, Centroid, DataPoint> {

	private final List<Centroid> centers = new ArrayList<>();
	private DistanceMeasurer distanceMeasurer;

	/**
	 *
	 * In this method, all centroids are loaded into memory as, in map(), we are going to compute the distance
	 * (similarity) of the data point with all centroids and associate the data point with its nearest centroid.
	 * Note that we load the centroid file on our own, which is not the same file as the one that hadoop loads in map().
	 *
	 *
	 * @param context Think of it as a shared data bundle between the main class, mapper class and the reducer class.
	 *                One can put something into the bundle in KMeansClusteringJob.class and retrieve it from there.
	 *
	 */
    @SuppressWarnings("deprecation")
	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		super.setup(context);

		// We get the URI to the centroid file on hadoop file system (not local fs!).
		// The url is set beforehand in KMeansClusteringJob#main.
		Configuration conf = context.getConfiguration();
		Path centroids = new Path(conf.get("centroid.path"));
		FileSystem fs = FileSystem.get(conf);

		// After having the location of the file containing all centroids data,
		// we read them using SequenceFile.Reader, which is another API provided by hadoop for reading binary file
		// The data is modeled in Centroid.class and stored in global variable centers, which will be used in map()
		try (SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf)) {
			Centroid key = new Centroid();
			IntWritable value = new IntWritable();
			int index = 0;
			while (reader.next(key, value)) {
				Centroid centroid = new Centroid(key);
				centroid.setClusterIndex(index++);
				centers.add(centroid);
			}
		}

		// This is for calculating the distance between a point and another (centroid is essentially a point).
		distanceMeasurer = new EuclidianDistance();
	}

	/**
	 *
	 * After everything is ready, we calculate and re-group each data-point with its nearest centroid,
	 * and pass the pair to reducer.
	 *
	 * @param centroid key
	 * @param dataPoint value
	 */
	@Override
	protected void map(Centroid centroid, DataPoint dataPoint, Context context) throws IOException,
			InterruptedException {

		Centroid nearest = null;
		double nearestDistance = Double.MAX_VALUE;

		for (Centroid c : centers) {
			double dist = distanceMeasurer.measureDistance(c.getCenterVector(), dataPoint.getVector());
			if (nearest == null) {
				nearest = c;
				nearestDistance = dist;
			} else {
				if (nearestDistance > dist) {
					nearest = c;
					nearestDistance = dist;
				}
			}
		}
		context.write(nearest, dataPoint);
	}

}
