/*---==>Custom FIle Upload--==>*/
$(".file-upload").each(function() {

  var fI  = $(this).children('input'), 
      btn = $(this).children('.btn-upload'), 
      i1  = $(this).children('.i-pic-upload'),
    i2  = $(this).children('.i-deselect'), 
    dN  = "No file Selected", 
    tfN = $(this).find('.text-file-name'), 
    bT  = $(this).find('.text-browse');
    bT.text('Browse...');
    tfN.text('No file Selected');

    btn.click(function(b) {
       b.preventDefault(); 
       fI.click();
    });
    function readURL(input) {
      if (input.files && input.files[0]) {
        var reader = new FileReader();
          reader.onload = function(e) {
          i1.css("background", "url("+e.target.result+") no-repeat center").removeClass('fa fa-image');
        }
        reader.readAsDataURL(input.files[0]);
      }
    }
  fI.change(function(e) {
    readURL(this); 
    var fN = e.target.files[0].name; 
    tfN.text(fN); 
    i2.fadeIn(200); 
    bT.text('Change...');
  });
  i2.click(function() {
    i2.fadeOut(100);
    window.setTimeout(function() {
      i1.css("background", "#ebebeb").addClass('fa fa-image');
      bT.text('Browse...');
      tfN.text('No file Selected');
      fI.val(null);
    }, 200);

  });
});

const btn = document.querySelector(".btn-submit");
console.log(btn);
const preloader = document.querySelector(".preloader");
preloader.classList.add("hidden");

function load() {
  preloader.classList.remove("hidden");
}

// const imgPath = document.getElementById("s-pic");

const reader = new FileReader();

reader.addEventListener("load", function () {
    // convert image file to base64 string and save to localStorage
    localStorage.setItem("image", reader.result);
}, false);




function save_pic() {
    const imgPath = document.getElementById("s-pic").files[0];
    if (imgPath) {
      console.log(imgPath);
      reader.readAsDataURL(imgPath);
    }
}

btn.addEventListener("click", () => {
  preloader.classList.remove("hidden");
})





function smaller() {

  const icon = document.getElementsByClassName('fa-image')[0];
  console.log(icon)
  if(window.innerWidth <700) {
    icon.setAttribute('style', "font-size:1.5em");
  }
  else {
    icon.setAttribute('style', "font-size:2em");
  }
}

$(window).resize(smaller);

smaller();