using UnityEngine;
using Unity.Sentis;
using System.IO;

public class LocalInference : MonoBehaviour
{
    private int currentIndex = 0;
    private float[][] data;

    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;

    public GameObject pred_hand;

    void Start()
    {
        // read from csv file
        string path = "Assets/Scripts/2_rear.csv";
        string[] lines = File.ReadAllLines(path);

        data = new float[lines.Length][];

        for (int i = 0; i < lines.Length; i++)
        {
            string[] values = lines[i].Split(',');

            data[i] = new float[values.Length];

            for (int j = 0; j < 50; j++)
            {
                data[i][j] = float.Parse(values[j]);
            }
        }

        // initialize Unity Sentis with the ONNX model
        if (modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);

            worker.Schedule(new Tensor<float>(new TensorShape(1,50,24)));
        }
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (data != null && data.Length > 0)
        {
            // update the actual hand position
            this.gameObject.transform.position = new Vector3(0,0,data[currentIndex][49]);
            currentIndex = (currentIndex + 1) % data.Length;

            //// Inference the lstm model to predict hand position 200ms ahead
            // 1. create input tensor from the csv file

            var input_seq_len = currentIndex > 50 ? 50 : currentIndex;
            Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, input_seq_len, 24));

            for (int i = 0; i < input_seq_len; i++)
            {
                for (int j = 0; j < 24; j++)
                {
                    inputTensor[0, i, j] = data[currentIndex - input_seq_len + i][j];
                }
            }

            // 2. schedule the input tensor

            worker.Schedule(inputTensor);
            var outputTensor = worker.PeekOutput() as Tensor<float>;
            var output = outputTensor.DownloadToArray();

            // 3. update the hand position

            // sum the output(velocity) to result the position
            float sum = 0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += output[i];
            }

            // update the predicted hand position
            pred_hand.transform.position = new Vector3(0, 0, data[currentIndex][49] + sum);
        }
    }
}
