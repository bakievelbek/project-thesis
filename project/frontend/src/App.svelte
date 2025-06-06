

<script>
  import Enhancement from './routes/enhancement.svelte';
  import AddNoise from './routes/add-noise.svelte';
  let page = 'enhancement';
</script>

<nav>
  <button on:click={() => page = 'enhancement'}
    class:selected={page === 'enhancement'}>
    Enhancement
  </button>
  <button on:click={() => page = 'add-noise'}
    class:selected={page === 'add-noise'}>
    Add noise
  </button>
</nav>

{#if page === 'enhancement'}
  <Enhancement />
{:else if page === 'add-noise'}
  <AddNoise />
{/if}

<style>
  nav {
    margin-bottom: 1.5rem;
    display: flex;
    gap: 1rem;
    justify-content: center;
  }
  button.selected {
    font-weight: bold;
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
    background: none;
    border: none;
    cursor: pointer;
  }
  button {
    padding: 0.5em 1.5em;
    background: none;
    border: none;
    font-size: 1.1em;
    cursor: pointer;
  }
</style>
