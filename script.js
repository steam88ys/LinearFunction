/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    let rawData = [];
    for(let x=0; x<25; x++)
    {
        let obj = {
            "x" : x,
            "y" : 2 * x + 1
        }
        rawData.push(obj);
    }
    return rawData;
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // hidden layer 추가
    model.add(tf.layers.dense({units: 1, useBias: true}));

    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));

    return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * y on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.x)
        const labels = data.map(d => d.y);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        return {
            inputs: inputTensor,
            labels: labelTensor
        }
    });
}

async function trainModel(model, inputs, labels, epochs) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function testModel(model, inputData, normalizationData, epochs) {

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {

        let xs = [];
        for(let x=0; x<30; x+=0.5)
        {
            xs.push(x);
        }

        const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
        const preds = model.predict(xsTensor);

        // Un-normalize the data
        return [xsTensor.dataSync(), preds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    });

    const originalPoints = inputData.map(d => ({
        x: d.x, y: d.y,
    }));

    tfvis.render.scatterplot(
        {name: '예측데이터 확인 학습횟수: ' + epochs},
        {values: [originalPoints, predictedPoints], series: ['학습', '예측']},
        {
            xLabel: 'x',
            yLabel: 'y',
            height: 300
        }
    );
}

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.x,
        y: d.y,
    }));

    tfvis.render.scatterplot(
        {name: 'x v y'},
        {values},
        {
            xLabel: 'x',
            yLabel: 'y',
            height: 300
        }
    );

    // More code will be added below
    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels, 100);
    testModel(model, data, tensorData, 100);

    await trainModel(model, inputs, labels, 100);
    testModel(model, data, tensorData, 100+100);

    await trainModel(model, inputs, labels, 300);
    testModel(model, data, tensorData, 100+100+300);
}

document.addEventListener('DOMContentLoaded', run);