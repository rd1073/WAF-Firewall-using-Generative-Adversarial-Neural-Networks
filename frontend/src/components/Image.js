import React, { useState } from 'react';
import axios from 'axios';

function Imagee() {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);

  const handleChange = (e) => {
    const files = Array.from(e.target.files).slice(0, 20);
    setImages(files);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    images.forEach(image => formData.append('images', image));

    try {
      const res = await axios.post('http://localhost:5000/upload', formData);
      setResults(res.data);
    } catch (error) {
      alert(error.response?.data?.error || "Error uploading images");
    }
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h2>Upload Images for Text Extraction</h2>
      <input type="file" multiple accept="image/*" onChange={handleChange} />
      <button onClick={handleUpload}>Upload and Extract</button>

      <div>
        {results.map((result, idx) => (
          <div key={idx} style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #ccc' }}>
            <h4>Image {result.image_number}</h4>
            <pre>{result.text}</pre>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Imagee;
