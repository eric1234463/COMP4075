import csvjson from "csvjson";
import fs from "fs";
import path from "path";
import brain from 'brain.js';

const net = new brain.NeuralNetwork();

const data = fs.readFileSync(path.join(__dirname, "airbnb.csv"), {
  encoding: "utf8"
});

const options = {
  delimiter: ",", // optional
  quote: '"' // optional
};

const dataArr = csvjson.toObject(data, options);

const modal = dataArr.map(element => {
  const modalData = {
    input: {
      accommodates: element.accommodates,
      bedrooms: element.bedrooms,
      minstay: element.minstay
    },
    output: {
      price: element.price
    }
  }
  return modalData;
});

console.log(modal);
net.train(modal);
var output = net.run({ accommodates: 1, bedrooms: 1, minstay: 1 });
console.log(output);