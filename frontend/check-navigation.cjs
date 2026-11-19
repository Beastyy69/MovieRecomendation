// Run against an isolated Vite instance with Playwright Core and installed Edge.
const {chromium}=require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
  const page=await browser.newPage();
  const errors=[];
  page.on('pageerror',err=>errors.push(err.message));
  await page.goto(process.env.TEST_URL||'http://localhost:5173');
  await page.getByRole('link',{name:'Explore film'}).click();
  const avatar=page.url();
  await page.getByRole('heading',{level:1,name:'Avatar',exact:true}).waitFor();
  const avatarArt=await page.locator('header:visible img').getAttribute('src');
  await page.goBack();
  await page.getByRole('searchbox').fill('Thor: Ragnarok');
  await page.locator('.results a').first().click();
  const thor=page.url();
  assert.notEqual(avatar,thor);
  await page.getByRole('heading',{name:'More like Thor: Ragnarok',exact:true}).waitFor();
  await page.locator('.story p').first().waitFor();
  assert.ok((await page.locator('.story p').first().textContent()).length>10);
  assert.notEqual(await page.locator('header:visible img').getAttribute('src'),avatarArt);
  await page.locator('.shelf:visible a').first().waitFor();
  await page.getByRole('button',{name:'Watch trailer'}).click();
  await page.getByRole('dialog',{name:'Thor: Ragnarok trailer'}).waitFor();
  await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'♡ Like'}).click();
  await page.getByRole('button',{name:'♥ Liked'}).waitFor();
  await page.getByRole('button',{name:'Mark seen'}).click();
  await page.getByRole('button',{name:'✓ Seen'}).waitFor();
  await page.reload();
  await page.getByRole('button',{name:'✓ Seen'}).waitFor();
  await page.getByRole('button',{name:'♥ Liked'}).waitFor();
  await page.goBack();
  assert.equal(await page.getByRole('searchbox').inputValue(),'Thor: Ragnarok');
  await page.goForward();
  assert.equal(page.url(),thor);
  await page.setViewportSize({width:390,height:844});
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.route('**/catalog/*/media',async route=>{
    await new Promise(r=>setTimeout(r,700));
    await route.fulfill({json:{backdrop_url:null,trailer_url:null}}).catch(()=>{});
  });
  await page.reload();
  await page.getByRole('heading',{level:1,name:'Thor: Ragnarok'}).waitFor();
  assert.equal(await page.locator('header:visible img').evaluate(img=>getComputedStyle(img).animationName),'none');
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.goBack();await page.goForward();
  await page.waitForTimeout(900);
  assert.equal(await page.locator('header:visible img').getAttribute('src'),await page.getByRole('img', {name:'Thor: Ragnarok poster', exact:true}).getAttribute('src'));
  await page.keyboard.press('Tab');
  assert.ok(await page.locator(':focus').count());
  assert.deepEqual(errors,[]);
  console.log('PASS Avatar',avatar,'Thor',thor,'navigation, synopsis, shelf, modal, saved preferences, history, fallback and mobile reduced motion');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
