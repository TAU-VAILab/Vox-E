$(document).ready(function() {
  $(".result-video").on("mouseover", function(event) {
    this.play();

  }).on('mouseout', function(event) {
    this.pause();

  });
})

$("#myvideo").hover(function(event) {
  if(event.type === "mouseenter") {
      $(this).attr("controls", "");
  } else if(event.type === "mouseleave") {
      $(this).removeAttr("controls");
  }
});