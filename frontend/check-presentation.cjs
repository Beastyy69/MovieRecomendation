const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.env.TEST_URL || 'http://127.0.0.1:5187';
    await page.route('**/api/discovery', async route => {
      await new Promise(resolve => setTimeout(resolve, 900));
      await route.continue();
    });
    await page.goto(base);
    await page.getByRole('status', {name:'Loading CineSense'}).waitFor();
    await page.getByRole('link', {name:'Explore film'}).waitFor();
    assert.equal(await page.locator('.preloader').count(), 0);
    assert.equal(await page.locator('.spotlight-card').count(), 1);
    await page.goto(base + '/missing-page');
    await page.getByRole('heading', {name:'This scene is missing.'}).waitFor();
    await page.getByRole('link', {name:'Back to discovery'}).click();
    await page.getByRole('link', {name:'Explore film'}).waitFor();
    await page.goto(base + '/movie/not-a-film');
    await page.getByRole('heading', {name:'This scene is missing.'}).waitFor();
    await page.emulateMedia({reducedMotion:'reduce'});
    assert.equal(await page.locator('main').evaluate(el=>getComputedStyle(el,'::before').animationName), 'none');
    await page.route('**/api/discovery', route=>route.fulfill({status:503,json:{detail:'Unavailable'}}));
    await page.goto(base);
    await page.getByText('Could not load the film catalog.').waitFor();
    await page.unroute('**/api/discovery');
    await page.getByRole('button', {name:'Retry'}).click();
    await page.getByRole('link', {name:'Explore film'}).waitFor();
    assert.deepEqual(errors, []);
    console.log('PASS preloader synchronization, React Bits integration, route/movie 404, recovery and reduced motion');
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
