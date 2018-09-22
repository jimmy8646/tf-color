let data
let colors = []
let xs, ys
let model
let labels = []
let rSlider, gSlider, bSlider
let labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish',
]


function preload() {
  data = loadJSON('color.json')
}

function setup() {
  rSlider = createSlider(0, 255, 255);
  gSlider = createSlider(0, 255, 0);
  bSlider = createSlider(0, 255, 255);
  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255] //Normalization
    colors.push(col)
    labels.push(labelList.indexOf(record.label))
  }
  xs = tf.tensor2d(colors)
  // console.log(xs.shape) //[row,col]

  let labelsTensor = tf.tensor1d(labels, 'int32')
  // labelsTensor.print()

  ys = tf.oneHot(labelsTensor, 9)

  model = tf.sequential()

  let hiddenLayer = tf.layers.dense({
    units: 32,
    inputDim: 3, // R,G,B 三種 input
    activation: 'sigmoid',
  })
  let outputLayer = tf.layers.dense({
    units: 9, // 9 種 label
    activation: 'softmax',
  })

  model.add(hiddenLayer)
  model.add(outputLayer)

  const learnRate = 0.2
  const optimizer = tf.train.sgd(learnRate)

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy'
  })

  train().then((res) => {
    console.log(res.history.loss)
  })
}

async function train() {

  const options = {
    epochs: 10,
    validationSplit: 0.1, // 10%
    shuffle: true,
    callbacks: {
      onTrainBegin: () => console.log('onTrainBegin'),
      onTrainEnd: () => console.log('onTrainEnd'),
      onEpochBegin: () => console.log('onEpochBegin'),
      onEpochEnd: (num, logs) => {
        console.log(`epochs: ${num} 結束， Loss值: ${logs.val_loss}`)
        // console.log(logs)
      },
      onBatchBegin: () => console.log('onBatchBegin'),
      onBatchEnd: () => {
        // console.log('onBatchEnd')
        return tf.nextFrame()
      }
    }
  }

  return await model.fit(xs, ys, options)
}

function draw() {
  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();
  background(r, g, b);
  strokeWeight(2);
  stroke(255);
  line(frameCount % width, 0, frameCount % width, height);

  const xs = tf.tensor2d([
    [r / 255, g / 255, b / 255]
  ])

  let result = model.predict(xs)
  let index = result.argMax(1).dataSync() //拿取機率最高的index

  let label = labelList[index]
  console.log(label)
  // result.print()
}