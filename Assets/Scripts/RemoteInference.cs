using UnityEngine;
using System.IO;
using Unity.Sentis;
using UnityEngine.Networking;
using System.Collections;

public class RemoteInference : MonoBehaviour
{
    private int currentIndex = 0;
    private float[][] data;

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
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (data != null && data.Length > 0)
        {
            // update the actual hand position
            this.gameObject.transform.position = new Vector3(0, 0, data[currentIndex][49]);
            currentIndex = (currentIndex + 1) % data.Length;

            // Inference the lstm model to predict hand position 200ms ahead
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



            // send inputTensor to server with http
            StartCoroutine(SendTensorToServer(inputTensor));
        }
    }

    private IEnumerator SendTensorToServer(Tensor<float> inputTensor)
    {
        string url = "http://127.0.0.1:5000";
        float[] tensorArray = inputTensor.DownloadToArray();
        float[][] tensor2DArray = ConvertTo2DArray(tensorArray, inputTensor.shape);

        // 2d array into json
        // need to be in the format of {"tensor": float2DArray}
        // no default json serializer in Unity, so we need to do it manually
        // I'm sorry but, do it yourself if needed
        var jsonData = "{}";

        using (UnityWebRequest www = UnityWebRequest.PostWwwForm(url, jsonData))
        {
            www.SetRequestHeader("Content-Type", "application/json");
            www.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(jsonData));
            www.downloadHandler = new DownloadHandlerBuffer();

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError(www.error);
            }
            else
            {
                // handle the response
                string responseText = www.downloadHandler.text;
                float[] output = JsonUtility.FromJson<float[]>(responseText);

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

    private float[][] ConvertTo2DArray(float[] tensorArray, TensorShape shape)
    {
        int rows = shape[1];
        int cols = shape[2];
        float[][] result = new float[rows][];

        for (int i = 0; i < rows; i++)
        {
            result[i] = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                result[i][j] = tensorArray[i * cols + j];
            }
        }

        return result;
    }
}
