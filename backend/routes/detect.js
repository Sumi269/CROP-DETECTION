const express = require('express');
const multer = require('multer');
const path = require('path');
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');
const History = require('../models/History');
const { authMiddleware } = require('../middleware/auth');

const router = express.Router();

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});
const upload = multer({ storage: storage });

router.post('/', authMiddleware, upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });

    // 1. Send image to Python Microservice (Runs on port 5001 now)
    const filePath = path.join(__dirname, '..', req.file.path);
    const formData = new FormData();
    formData.append('image', fs.createReadStream(filePath));

    // Calling Python service
    let aiResult;
    try {
        const response = await axios.post('http://localhost:5001/predict', formData, {
            headers: { ...formData.getHeaders() }
        });
        aiResult = response.data;
    } catch (apiErr) {
        console.error("Python API Error:", apiErr.message);
        // Fallback or Fail gracefully
        return res.status(502).json({ error: "AI Service is offline or failed." });
    }

    // 2. Determine solution from AI Result (Python side provides crop, status, confidence, full_label)
    // We can fetch a solution from MongoDB, or construct it basically.
    // For now, use the result passed by python or default string
    const solutionText = aiResult.status === "HEALTHY" ? "Your crop is healthy! Maintain regular care." : "Immediate action required. Apply appropriate fungicides and remove infected leaves.";

    // 3. Save to History
    const historyEntry = new History({
      user_id: req.user.id,
      image_url: `/uploads/${req.file.filename}`,
      crop: aiResult.crop,
      disease: aiResult.full_label || "Unknown",
      confidence: aiResult.confidence,
      solution: solutionText
    });
    await historyEntry.save();

    // 4. Return to frontend
    res.json({
        crop: aiResult.crop,
        disease: aiResult.full_label,
        confidence: aiResult.confidence,
        solution: solutionText,
        image_url: historyEntry.image_url
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
