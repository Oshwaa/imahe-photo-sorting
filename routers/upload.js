const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
//set up storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    cb(
      null,
      file.fieldname + "-" + Date.now() + path.extname(file.originalname)
    );
  },
});
// To filter uploads
const upload = multer({
  storage: storage,
  fileFilter: function (req, file, cb) {
    const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type"), false);
    }
  },
});
//UPLOAD ROUTE
router.post("/upload", upload.array("photos"), (req, res) => {
  if (!req.files) {
    return res.status(400).json({ error: "No files were uploaded." });
  }
  const images = req.files;
  console.log("Uploaded files:", images);
  res.send("Files uploaded successfully");
});

router.get("/test", (req, res) => {
  res.json("Hello");
});
//HANDLE ERROR
router.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    // Multer-specific errors
    return res.status(400).send({ error: "Multer error occurred." });
  } else if (err) {
    // General errors
    return res.status(400).send({ error: err.message });
  }
  next();
});

module.exports = router;
