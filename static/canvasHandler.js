// Set up the canvas
console.log("sa from canvas handler");
var canvas = document.getElementById("crop-canvas");
var canvasImg = document.getElementById("canvas-img");
try{
  var ctx = canvas.getContext("2d");
  ctx.strokeStyle = "#222222";
  ctx.lineWith = 2;
  canvas.width = canvasImg.clientWidth;
  canvas.height = canvasImg.clientHeight;
  dotArray = new Array([0, 0]);
  dotArray.pop([0,0])

ctx.drawImage(canvasImg, 0, 0,canvas.width,canvas.height);

// Set up mouse events for drawing
var drawing = false;
var mousePos = { x:0, y:0 };
var lastPos = mousePos;
canvas.addEventListener("mousedown", function (e) {
        drawing = true;
  lastPos = getMousePos(canvas, e);
}, false);
canvas.addEventListener("mouseup", function (e) {
  sendData();
  drawing = false;
}, false);
canvas.addEventListener("mousemove", function (e) {
  mousePos = getMousePos(canvas, e);
}, false);

// after selection end send data to server
function sendData() {
  axios({
    method: "get",
    url:"crop",
    headers: { "Content-Type": "multipart/form-data",
              "points": dotArray,
              "width": canvas.width,
            }
  })
    .then(function (response) {
      window.location.reload();
    })
}
// Get the position of the mouse relative to the canvas
function getMousePos(canvasDom, mouseEvent) {
  var rect = canvasDom.getBoundingClientRect();
  return {
    x: mouseEvent.clientX - rect.left,
    y: mouseEvent.clientY - rect.top
  };
}

// Get a regular interval for drawing to the screen
window.requestAnimFrame = (function (callback) {
        return window.requestAnimationFrame || 
           window.webkitRequestAnimationFrame ||
           window.mozRequestAnimationFrame ||
           window.oRequestAnimationFrame ||
           window.msRequestAnimaitonFrame ||
           function (callback) {
        window.setTimeout(callback, 1000/60);
           };
})();

// Draw to the canvas
function renderCanvas() {
  if (drawing) {
    ctx.moveTo(lastPos.x, lastPos.y);
    ctx.lineTo(mousePos.x, mousePos.y);
    newDot = new Array(lastPos.x,lastPos.y);
    dotArray.push(newDot);
    newDot = new Array(mousePos.x,mousePos.y);
    dotArray.push(newDot);
    ctx.stroke();
    lastPos = mousePos;
  }
}

// Allow for animation
(function drawLoop () {
  requestAnimFrame(drawLoop);
  renderCanvas();
})();

// Set up touch events for mobile, etc
canvas.addEventListener("touchstart", function (e) {
        mousePos = getTouchPos(canvas, e);
  var touch = e.touches[0];
  var mouseEvent = new MouseEvent("mousedown", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
}, false);
canvas.addEventListener("touchend", function (e) {
  var mouseEvent = new MouseEvent("mouseup", {});
  canvas.dispatchEvent(mouseEvent);
}, false);
canvas.addEventListener("touchmove", function (e) {
  var touch = e.touches[0];
  var mouseEvent = new MouseEvent("mousemove", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
}, false);

// Get the position of a touch relative to the canvas
function getTouchPos(canvasDom, touchEvent) {
  var rect = canvasDom.getBoundingClientRect();
  return {
    x: touchEvent.touches[0].clientX - rect.left,
    y: touchEvent.touches[0].clientY - rect.top
  };
}

// Prevent scrolling when touching the canvas
document.body.addEventListener("touchstart", function (e) {
  if (e.target == canvas) {
    e.preventDefault();
  }
}, false);
document.body.addEventListener("touchend", function (e) {
  if (e.target == canvas) {
    e.preventDefault();
  }
}, false);
document.body.addEventListener("touchmove", function (e) {
  if (e.target == canvas) {
    e.preventDefault();
  }
}, false);
}
catch(err)
{

}
