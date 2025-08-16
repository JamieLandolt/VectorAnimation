window.addEventListener("load", () => {
    // CANVAS /////////////////////////////////////////////////////////////////////////////////////
    const canvas = document.querySelector("#canvas");
    const context = canvas.getContext("2d");

    let painting = false;
    let points = [];

    function clearCanvas() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        points = [];
    }   

    function startPosition(e) {
        painting = true;
        context.beginPath();
        clearCanvas();
        const rect = canvas.getBoundingClientRect();
        context.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    } 

    function finishPosition() { 
        painting = false; 
        context.beginPath();
        console.log("Points", points);
    }

    function draw(e) {
        if (!painting) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        context.lineWidth = 2;
        context.lineCap = "round";
        context.lineTo(x, y);
        context.stroke();
        context.beginPath();
        context.moveTo(x, y);
        
        // Add to points array
        points.push({
            x: x,
            y: y,
            timestamp: Date.now()
        });
    }

    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("mouseup", finishPosition);
    canvas.addEventListener("mousemove", draw);

    // CLEAR BUTTON //////////////////////////////////////////////////////////////////////////////
    const clearBtn = document.querySelector("#clear-btn");

    clearBtn.addEventListener("click", clearCanvas)
})

