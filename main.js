let data
let colors = []
let xs,ys
let labels = []
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

  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255] //Normalization
    colors.push(col)
    labels.push(labelList.indexOf(record.label))
  }
  xs = tf.tensor2d(colors)
  // console.log(xs.shape) //[row,col]

  let labelsTensor = tf.tensor1d(labels,'int32') 
  // labelsTensor.print()

  ys = tf.oneHot(labelsTensor,9)
  
}

function draw() {

}