let data
let colors = []
let xs

function preload() {
  data = loadJSON('color.json')
}

function setup() {

  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255] //標準化
    colors.push(col)
  }
  xs = tf.tensor2d(colors)
  // console.log(xs.shape) [row,col]
}

function draw() {

}