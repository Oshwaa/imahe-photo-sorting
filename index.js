const express = require("express");
const app = express();
const uploadRoute = require("./routers/upload");

const port = 3000;

app.use("/api", uploadRoute);

app.get("/", (req, res) => {
  res.send("Hello World");
});

console.log(`Listening to PORT ${port}`);
app.listen(port);
