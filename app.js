import csvjson from "csvjson";
import fs from "fs";
import path from "path";
import DecisionTree from "decision-tree";
import express from "express";
import prompt from "prompt";
import KNN from "ml-knn";

//render app
const app = express();
app.set('view engine', 'ejs');  
app.use(express.static('public'));
app.get("/", (req, res) => {
  const data = getData();
  const knn = convertKnnModel(data);
  res.render('index', { data: data })
});
app.listen(3000, () => console.log("Example app listening on port 3000!"));

//DT
const getData = () => {
  const data = fs.readFileSync(path.join(__dirname, "airbnb.csv"), {
    encoding: "utf8"
  });

  const options = {
    delimiter: ",", // optional
    quote: '"' // optional
  };

  const dataArr = csvjson.toObject(data, options);
  return dataArr;
};

const callDT = dataArr => {
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
};

const convertKnnModel = dataArr => {
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
}

//k nearest-neighbor

let knn;
let seperationSize; // To seperate training and test data
let X = [], y = [];
let trainingSetX = [], trainingSetY = [], testSetX = [], testSetY = [];

let kNN_modal = getData().map(element => ({
  room_type: element.room_type,
  accommodates: parseInt(element.accommodates),
  bedrooms: parseInt(element.bedrooms),
  minstay: parseInt(element.minstay),
  price: parseInt(element.price)
}));

  seperationSize = 0.7 * kNN_modal.length;
  kNN_modal = shuffleArray(kNN_modal);
  dressData();

function dressData() {

    let types = new Set();

    kNN_modal.forEach((row) => {
        types.add(row.room_type);
    });

    let typesArray = [...types];
    
    // typesArray:
    // entire home = 0
    // private room = 1
    // shared room = 2

    kNN_modal.forEach((row) => {
        let rowArray, typeNumber;

        rowArray = Object.keys(row).map(key => parseFloat(row[key])).slice(1, 5);

        typeNumber = typesArray.indexOf(row.room_type); // Convert type(String) to type(Number)

        X.push(rowArray);
        y.push(typeNumber);
    });

    trainingSetX = X.slice(0, seperationSize);
    trainingSetY = y.slice(0, seperationSize);

    testSetX = X.slice(seperationSize);
    testSetY = y.slice(seperationSize);

    train();
}

function train() {
    knn = new KNN(trainingSetX, trainingSetY, {k: 7});
    test();
}

function test() {
    const result = knn.predict(testSetX);
    const testSetLength = testSetX.length;
    const predictionError = error(result, testSetY);
    console.log(`Test Set Size = ${testSetLength} and number of Misclassifications = ${predictionError}`);
    predict();
}

function error(predicted, expected) {
    let misclassifications = 0;
    for (var index = 0; index < predicted.length; index++) {
        if (predicted[index] !== expected[index]) {
            misclassifications++;
        }
    }
    return misclassifications;
}

function predict() {
    let temp = [];
    prompt.start();
    
    prompt.get(['accommodates', 'bedrooms', 'minstay', 'price'], function (err, result) {
        if (!err) {
            for (var key in result) {
                temp.push(parseFloat(result[key]));
            }
            console.log(`With ${temp} -- room_type =  ${knn.predict(temp)}`);
        }
    });
}

function shuffleArray(array) {
  //for random array order
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}
