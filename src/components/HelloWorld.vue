<template>
  <div class="hello">
    <div><img width="570" height="380" src="/84*84.png" /></div>
    <div><canvas id="canvas" width="570" height="380"></canvas></div>
    <div><canvas id="canvas2"></canvas></div>
  </div>
</template>

<script>
import { gaussBlur } from './gaussBlur';

export default {
  async mounted() {
    gaussBlur();
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.hello {
  /* background: lightblue; */
}
canvas {
  border: 5px solid #ccc;
}
</style>
