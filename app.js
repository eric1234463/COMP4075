import csvjson from "csvjson";
import fs from "fs";
import path from "path";
import DecisionTree from "decision-tree";
import nn from "nearest-neighbor";
import express from 'express';

const app = express()

app.get('/', (req, res) => res.send('Hello World!'))

app.listen(3000, () => console.log('Example app listening on port 3000!'));

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
  };
  return modalData;
});

// console.log(modal);

// net.train(modal);
// var output = net.run({ accommodates: 1, bedrooms: 1, minstay: 1 });
// console.log(output);

//k nearest-neighbor
const kNN_modal = dataArr.map(element => ({
  room_type: element.room_type,
  neighborhood: element.neighborhood,
  accommodates: parseInt(element.accommodates),
  bedrooms: parseInt(element.bedrooms),
  minstay: parseInt(element.minstay),
  price: parseInt(element.price)
}));

var query = {
  room_type: "Shared room",
  neighborhood: null,
  accommodates: 1,
  bedrooms: 2,
  minstay: 1,
  price: null
};

var fields = [
  { name: "room_type", measure: nn.comparisonMethods.word },
  { name: "neighborhood", measure: nn.comparisonMethods.word },
  { name: "accommodates", measure: nn.comparisonMethods.number, max: 100 },
  { name: "bedrooms", measure: nn.comparisonMethods.number, max: 100 },
  { name: "minstay", measure: nn.comparisonMethods.number, max: 100 },
  { name: "price", measure: nn.comparisonMethods.number, max: 100 }
];

nn.findMostSimilar(query, kNN_modal, fields, (nearestNeighbor, probability) => {
  console.log(query);
  console.log(nearestNeighbor);
  console.log(probability);
});

// console.log(kNN_modal);

const training_data = dataArr.map(element => ({
  accommodates: parseInt(element.accommodates),
  bedrooms: parseInt(element.bedrooms),
  price: parseInt(element.price)
}));

const features = ["bedrooms", "accommodates"];

const class_name = "price";

const dt = new DecisionTree(training_data, class_name, features);

const predicted_class = dt.predict({
  bedrooms: 1,
  accommodates: 2
});

// const test_data = [
//   { bedrooms: 1, accommodates: 1, price: 10 },
//   { bedrooms: 2, accommodates: 2, price: 27 },
//   { bedrooms: 3, accommodates: 3, price: 41 },
//   { bedrooms: 4, accommodates: 4, price: 55 }
// ];

console.log(predicted_class);
