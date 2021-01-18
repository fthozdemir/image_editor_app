const inpFile=document.getElementById("initialImage");
const grayishButton=document.getElementById("grayishButton");
const rotateButton=document.getElementById("rotateButton");
const cropButton=document.getElementById("cropButton");
const flipButton=document.getElementById("flipButton");
const mirrorButton=document.getElementById("mirrorButton");
const embossButton=document.getElementById("embossButton");
const inverseButton=document.getElementById("inverseButton");
const lumosButton=document.getElementById("lumosButton");
const pixelButton=document.getElementById("pixelButton");
const popButton=document.getElementById("popButton");
const oldtvButton=document.getElementById("oldtvButton");
const sketchButton=document.getElementById("sketchButton");
const splashButton=document.getElementById("splashButton");
const sepyaButton=document.getElementById("sepyaButton");
const cartoonButton=document.getElementById("cartoonButton");
const oilyButton=document.getElementById("oilyButton");
const autoConButton=document.getElementById("autoConButton");
const abstractButton=document.getElementById("abstractButton");
const balmyButton=document.getElementById("balmyButton");
const linesButton=document.getElementById("linesButton");
const blushButton=document.getElementById("blushButton");
const coldButton=document.getElementById("coldButton");
const glassButton=document.getElementById("glassButton");
const xproButton=document.getElementById("xproButton");
const daylightButton=document.getElementById("daylightButton");
const moonButton=document.getElementById("moonButton");
const blueishButton=document.getElementById("blueishButton");

console.log("sa from image handler");

$('#saveButton').click(function(){ saveButtonFunction(); return false; });

inpFile.addEventListener("change", function () {
  const myUrl=this.getAttribute("apiurl");
  const reader = new FileReader();
  const file = this.files[0];
  if (!file) {
    return;
  }
  reader.onload = function (e) {
    const bodyFormData = new FormData();
    const data = e.target.result.split(",", 2)[1];
    bodyFormData.set("image", data);
    

    axios({
      method: "post",
      data: bodyFormData,
      url:myUrl,
      headers: { "Content-Type": "multipart/form-data" }
    })
      .then(function (response) {
        window.location.reload();
      })

  };
  reader.readAsDataURL(file);
});


function saveButtonFunction() {
  console.log("save clicked")
  axios({
    method: "get",
    url:"save",
    headers: { "Content-Type": "multipart/form-data" }
  })
    .then(function (response) {
      window.location.reload();
    })

};

grayishButton.addEventListener("click", function () {
      axios({
        method: "get",
        url:"grayish",
        headers: { "Content-Type": "multipart/form-data" }
      })
        .then(function (response) {
          window.location.reload();
        })
  
    });

embossButton.addEventListener("click", function () {
      axios({
        method: "get",
        url:"emboss",
        headers: { "Content-Type": "multipart/form-data" }
      })
        .then(function (response) {
          window.location.reload();
        })
  
    });

pixelButton.addEventListener("click", function () {
      axios({
        method: "get",
        url:"pixel",
        headers: { "Content-Type": "multipart/form-data" }
      })
        .then(function (response) {
          window.location.reload();
        })
  
    });
popButton.addEventListener("click", function () {
      axios({
        method: "get",
        url:"pop",
        headers: { "Content-Type": "multipart/form-data" }
      })
        .then(function (response) {
          window.location.reload();
        })
  
    });

mirrorButton.addEventListener("click", function () {
      axios({
        method: "get",
        url:"mirror",
        headers: { "Content-Type": "multipart/form-data" }
      })
        .then(function (response) {
          window.location.reload();
        })
  
    });

inverseButton.addEventListener("click", function () {
      axios({
        method: "get",
        url:"inverse",
        headers: { "Content-Type": "multipart/form-data" }
      })
        .then(function (response) {
          window.location.reload();
        })
  
    });

rotateButton.addEventListener("click", function () {
      const myValue=this.querySelector("output").value;
        axios({
          method: "get",
          url:"rotate",
          headers: { "Content-Type": "multipart/form-data",
                     "angle":myValue }
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

lumosButton.addEventListener("click", function () {
        const myValue=this.querySelector("output").value;
          axios({
            method: "get",
            url:"lumos",
            headers: { "Content-Type": "multipart/form-data",
                       "lumen":myValue }
          })
            .then(function (response) {
              window.location.reload();
            })
      
        });

contrastButton.addEventListener("click", function () {
          const myValue=this.querySelector("output").value;
            axios({
              method: "get",
              url:"contrast",
              headers: { "Content-Type": "multipart/form-data",
                         "contrast":myValue }
            })
              .then(function (response) {
                window.location.reload();
              })
        
          });
  
cropButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"cropselect",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });



flipButton.addEventListener("click", function () {
  const hor=document.getElementById("hor-flip").checked;
  const ver=document.getElementById("ver-flip").checked;

  console.log(hor);

        axios({
          method: "get",
          url:"flip",
          headers: { "Content-Type": "multipart/form-data" ,
        "hor":hor,
        "ver":ver,}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

oldtvButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"oldtv",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

sketchButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"sketch",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

splashButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"splash",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

sepyaButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"sepya",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

cartoonButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"cartoon",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

oilyButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"oily",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

autoConButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"autocon",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

abstractButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"abstractify",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });


balmyButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"balmy",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

coldButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"cold",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });
linesButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"lines",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });
blushButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"blush",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

glassButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"glass",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })
    
      });

xproButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"xpro",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })

      });

daylightButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"daylight",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })

      });

moonButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"moon",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })

      });

blueishButton.addEventListener("click", function () {
        axios({
          method: "get",
          url:"blueish",
          headers: { "Content-Type": "multipart/form-data"}
        })
          .then(function (response) {
            window.location.reload();
          })

      });