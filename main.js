let data
let colors = []
let xs, ys
let model
let labels = []
let rSlider, gSlider, bSlider
let labelList = [
  '紅色系',
  '綠色系',
  '藍色系',
  '橘色系',
  '黃色系',
  '粉色系',
  '紫色系',
  '棕色系',
  '灰色系',
]


function preload() {
  data = loadJSON('color.json')
}

function setup() {
  let canvas = createCanvas(300, 300).parent('canvas')
  rSlider = createSlider(0, 255, 255).parent('sliderR')
  gSlider = createSlider(0, 255, 0).parent('sliderG')
  bSlider = createSlider(0, 255, 0).parent('sliderB')
  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255] //Normalization
    colors.push(col)
    labels.push(labelList.indexOf(record.label))
  }
  xs = tf.tensor2d(colors)
  // console.log(xs.shape) //[row,col]

  let labelsTensor = tf.tensor1d(labels, 'int32')
  // labelsTensor.print()

  ys = tf.oneHot(labelsTensor, 9) // one hot encoding

  model = tf.sequential()

  let hiddenLayer = tf.layers.dense({
    units: 32, // 此隱藏層的神經元數量
    inputDim: 3, // R,G,B 三種 input
    activation: 'sigmoid', // 激活函數
  })
  let outputLayer = tf.layers.dense({
    units: 9, // 9 種 label
    activation: 'softmax',
  })

  model.add(hiddenLayer)
  model.add(outputLayer)

  const learnRate = 0.2
  const optimizer = tf.train.sgd(learnRate) // 梯度下降法

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy' // 優化 loss 演算法 for 分類問題
  })

  train().then((res) => {
    console.log(res.history.loss)
  })
}

async function train() {

  const options = {
    epochs: 30, // 輪迴訓練次數
    validationSplit: 0.1, // 10% 資料驗證
    shuffle: true, // 亂數取樣
    callbacks: {
      // onTrainBegin: () => console.log('onTrainBegin'),
      // onTrainEnd: () => console.log('onTrainEnd'),
      // onEpochBegin: () => console.log('onEpochBegin'),
      onEpochEnd: (num, logs) => {
        document.getElementById("status").innerHTML = `epochs: ${num} 結束， Loss值: ${logs.loss}`
        // console.log(`epochs: ${num} 結束， Loss值: ${logs.loss}`)
        // console.log(logs)
      },
      // onBatchBegin: () => console.log('onBatchBegin'),
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

  // 防止記憶體被吃光
  tf.tidy(() => {
    const xs = tf.tensor2d([
      [r / 255, g / 255, b / 255]
    ])

    let result = model.predict(xs)
    let index = result.argMax(1).dataSync() //拿取機率最高的 index
    let label = labelList[index]

    document.getElementById("result").innerHTML= label 
    // console.log(label)
    // console.log(tf.memory().numTensors)
    // result.print()

  })
  

}