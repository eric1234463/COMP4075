import csvjson from "csvjson";
import fs from "fs";
import path from "path";

var data = fs.readFileSync(path.join(__dirname, "airbnb.csv"), {
  encoding: "utf8"
});

var options = {
  delimiter: ",", // optional
  quote: '"' // optional
};

const object = csvjson.toObject(data, options);

console.log(object);