async function fetchSignal(){
  try{
    const res = await fetch('/api/signal');
    const j = await res.json();
    const s = document.getElementById('signal');
    const d = document.getElementById('details');
    s.innerHTML = `<h2>${j.signal}</h2><p>${j.reason || ''}</p>`;
    d.innerHTML = `<pre>${JSON.stringify(j, null, 2)}</pre>`;
  }catch(e){
    document.getElementById('signal').innerText = 'Error fetching signal';
    document.getElementById('details').innerText = e.toString();
  }
}

fetchSignal();
// auto-refresh every 60s
setInterval(fetchSignal, 60000);
