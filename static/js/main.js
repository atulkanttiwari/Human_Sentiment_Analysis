document.addEventListener('DOMContentLoaded', function(){
  const form = document.getElementById('predict-form');
  const btn = document.getElementById('predict-btn');
  const textarea = document.getElementById('text');
  const message = document.getElementById('message');

  if(!form) return;

  form.addEventListener('submit', function(e){
    if(!textarea.value.trim()){
      e.preventDefault();
      message.textContent = 'Please enter some text before predicting.';
      textarea.focus();
      return;
    }
    // show loading state
    if(btn){
      btn.disabled = true;
      btn.textContent = 'Predicting...';
    }
    message.textContent = 'Contacting model...';
  });
});
