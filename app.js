import csvjson from "csvjson";
import fs from "fs";
import path from "path";
import DecisionTree from "decision-tree";
import express from "express";
import KNN from "ml-knn";

//render app
const app = express();
app.set("view engine", "ejs");
app.use(express.static("public"));
app.get("/", (req, res) => {
  const data = getData();
  res.render("index", { data: data });
});

app.get("/result", (req, res) => {
  const inputData = {
    accommodates: req.query.accommodates,
    bedrooms: req.query.bedrooms
  };
  const price = callDT(inputData);
  const result = {
    price: price
  };
  res.render("result", { result: result, inputData: inputData });
});

app.get("/knn-result", (req, res) => {
  const inputData = {
    bedrooms: req.query.bedrooms,
    accommodates: req.query.accommodates,
    price: req.query.price
  };
  const roomType = predict(inputData);

  const mapping = {
    0: "Entire Home/apt",
    1: "Private Room",
    2: "Shared Room"
  };

  const result = {
    room_type: mapping[roomType]
  };
  res.render("knn-result", { result: result, inputData: inputData });
});

const port = process.env.PORT || 3000;

app.listen(port, () => console.log("App listening on port!", port));

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

const callDT = inputData => {

  const mapping = price => {
    if (price >= 1 && price <= 50) {
      return '1-50'
    } else if (price >= 51 && price <= 150) {
      return '51-150'
    } else {
      return '>150'
    }
  }

  const training_data = getData().map(element => ({
    accommodates: parseInt(element.accommodates),
    bedrooms: parseInt(element.bedrooms),
    price: mapping(element.price)
  }));

  const features = ["bedrooms", "accommodates"];

  const class_name = "price";

  const dt = new DecisionTree(training_data, class_name, features);

  const predicted_class = dt.predict(inputData);

  //increase performance
  // evaluation(training_data);
  return predicted_class;
};

const evaluation = training_data => {
  const seperationSize = 0.7 * training_data.length;

  let trainingSet = [],
    testingSet = [];

  trainingSet = training_data.slice(0, seperationSize);
  testingSet = training_data.slice(seperationSize);

  const features = ["bedrooms", "accommodates"];
  const class_name = "price";

  const dt = new DecisionTree(trainingSet, class_name, features);

  let misclassifications = 0;
  for (var index = 0; index < testingSet.length; index++) {
    let input = {
      accommodates: testingSet[index].accommodates,
      bedrooms: testingSet[index].bedrooms
    };
    let res = dt.predict(input);

    if (res !== testingSet[index].price) {
      misclassifications++;
    }
  }
  console.log('size:', index);
  console.log('DT:', misclassifications);
};

//k nearest-neighbor
let knn;
let seperationSize; // To seperate training and test data
let X = [],
  y = [];
let trainingSetX = [],
  trainingSetY = [],
  testSetX = [],
  testSetY = [];

let kNN_modal = getData().map(element => ({
  room_type: element.room_type,
  accommodates: parseInt(element.accommodates),
  bedrooms: parseInt(element.bedrooms),
  price: parseInt(element.price)
}));

seperationSize = 0.7 * kNN_modal.length;
kNN_modal = shuffleArray(kNN_modal);

const dressData = () => {
  let types = new Set();

  kNN_modal.forEach(row => {
    types.add(row.room_type);
  });

  let typesArray = [...types];

  // typesArray:
  // entire home = 0
  // private room = 1
  // shared room = 2

  kNN_modal.forEach(row => {
    let rowArray, typeNumber;

    rowArray = Object.keys(row)
      .map(key => parseFloat(row[key]))
      .slice(1, 4);

    typeNumber = typesArray.indexOf(row.room_type); // Convert type(String) to type(Number)

    X.push(rowArray);
    y.push(typeNumber);
  });

  trainingSetX = X.slice(0, seperationSize);
  trainingSetY = y.slice(0, seperationSize);

  testSetX = X.slice(seperationSize);
  testSetY = y.slice(seperationSize);

  train();
};

const train = () => {
  knn = new KNN(trainingSetX, trainingSetY, { k: 92 });
  test();
};

function test() {
  const result = knn.predict(testSetX);
  const testSetLength = testSetX.length;
  const predictionError = error(result, testSetY);
  console.log(
    `Test Set Size = ${testSetLength} and number of Misclassifications = ${predictionError}`
  );
  // predict();
}

dressData();

function error(predicted, expected) {
  let misclassifications = 0;
  for (var index = 0; index < predicted.length; index++) {
    if (predicted[index] !== expected[index]) {
      misclassifications++;
    }
  }
  return misclassifications;
}

const predict = inputData => {
  console.log(inputData);
  let temp = [];
  for (var key in inputData) {
    temp.push(parseInt(inputData[key]));
  }
  console.log(`With ${temp} -- room_type =  ${knn.predict(temp)}`);
  return knn.predict(temp);
};

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
