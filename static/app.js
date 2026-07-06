async function runPrediction() { // Function used by the html to get the prediction results, async to allow use of await
    //Uses getElementById to scour the HTML
    const smiles = document.getElementById("smiles-input").value; // Reads in the input typed by the user
    
    const resultEl = document.getElementById("result"); // Finds the area we will write the result to 
    resultEl.textContent = "Predicting..."; // Just lets the user know that progress is happening

    // Sends in the molecule into the API for prediction, and stores the result
    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json"},
        body: JSON.stringify({ smiles: smiles})
    });

    const data = await response.json(); // Turns the JSON response into a JS object, uses await to let other things happen while waiting

    resultEl.textContent = `Predicted logS: ${data.logS.toFixed(3)} (higher means more soluble in water)`; // Puts the results on screen
}