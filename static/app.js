async function runPrediction() { // Function used by the html to get the prediction results, async to allow use of await
    const raw = document.getElementById("smiles-input").value;
    const molecules = raw.split(",").map(s => s.trim()).filter(s => s.length > 0); // Turns a comma seperated input, into a list for a batch

    // Resets output areas
    const resultEl = document.getElementById("result");
    const measuredEl = document.getElementById("measured");
    const molImg = document.getElementById("mol-img");
  
    molImg.style.display = "none";
    resultEl.textContent = "";
    measuredEl.textContent = "";
    molImg.removeAttribute("src");     // clear the single-molecule image
    document.getElementById("batch-results").innerHTML = "";   // clear any old table

    if (molecules.length > 1) { // Runs if there is more than one molecule, and thus a batch
        try {
            await runBatch(molecules);
        } catch (e) {
            console.error(e);
            document.getElementById("result").textContent = "Batch failed — is the worker running?";
        }
        return; // Ends early to skip single input code
    }

    //Uses getElementById to scour the HTML
    const smiles = document.getElementById("smiles-input").value; // Reads in the input typed by the user
    
    resultEl.textContent = "Predicting..."; // Just lets the user know that progress is happening

    // Tries to fetch results from /predict in app.py
    try{
        // Sends in the molecule into the API for prediction, and stores the result
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json"},
            body: JSON.stringify({ smiles: smiles})
        });

        // If server returns an error, show the message and stop
        if(!response.ok){
            const err = await response.json();
            resultEl.textContent = `Error: ${err.detail}`;
            return;
        }

        const data = await response.json(); // Turns the JSON response into a JS object, uses await to let other things happen while waiting
        resultEl.textContent = `Predicted logS: ${data.logS.toFixed(3)} (higher means more soluble in water)`; // Puts the results on screen

        if(data.measured){
            const note = data.measured.in_training
                ? "Model was trained on this molecule"
                : "Used in validation, not trained on";
            measuredEl.textContent =`Measured logS: ${data.measured.measured.toFixed(3)} (${note})`;
        }
        else{
            measuredEl.textContent = "Not in dataset";
        }
        molImg.style.display = "";
        molImg.src = `/draw?smiles=${encodeURIComponent(smiles)}`; // Uses encodeURIComponent to escape special characters that may appear in SMILES

    } catch (e){ 
        console.error(e); // Logs the exact actual error
        resultEl.textContent = "Couldn't reach prediction service. Please check connection and try again"; // Gives something the user could do to try and fix problem
    }
}

async function pollJob(jobId) {
      while (true) {
          const resp = await fetch(`/jobs/${jobId}`);
          const job = await resp.json();
          if (job.status === "finished") return job.result;   // Return results array if done
          if (job.status === "failed") throw new Error("Batch job failed");
          await new Promise(r => setTimeout(r, 500));          // Wait and check back in 500 milliseconds 
      }
}


function renderTable(results) {
      let html = "<table border='1'><tr><th>Structure</th><th>Prediction</th></tr>"; // The start of the batch table, starts with header
      for (const r of results) { // Creates a row for every result
          if (r.error) {
              html += `<tr><td>—</td><td>${r.smiles}: ${r.error}</td></tr>`;
          } else {
              const img = `<img src="/draw?smiles=${encodeURIComponent(r.smiles)}" width="150">`;
              let text = `Predicted logS: ${r.logS.toFixed(3)}`;
              if (r.measured) {
                  const note = r.measured.in_training ? "trained on" : "held out — genuine test";
                  text += `<br>Measured: ${r.measured.measured.toFixed(3)} (${note})`;
              }
              html += `<tr><td>${img}</td><td>${text}</td></tr>`;
          }
      }
      html += "</table>";
      document.getElementById("batch-results").innerHTML = html;
  }

  async function runBatch(molecules) {
      document.getElementById("result").textContent = "Predicting batch...";
      const submit = await fetch("/predict/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles_list: molecules })
      });
      const { job_id } = await submit.json();
      const results = await pollJob(job_id);   // wait for the worker to finish
      document.getElementById("result").textContent = "";   // clear the "Predicting batch..." text
      renderTable(results);
  }

// Runs prediction on any form submittion whether button press or Enter key
document.getElementById("predict-form").addEventListener("submit", function(event){
    event.preventDefault(); // stops browser refresh
    runPrediction();
});