// API endpoint
const API_BASE = window.location.origin + '/api';

// Parse URL parameters
function getURLParams() {
    const params = new URLSearchParams(window.location.search);
    return {
        price: params.get('price') || '',
        skin_type: params.get('skin_type') || '',
        availability: params.get('availability') || ''
    };
}

// Display URL parameters in filters panel
function displayFilters() {
    const params = getURLParams();
    
    if (params.price) {
        document.getElementById('price-display').textContent = `${params.price}`;
    } else {
        document.getElementById('price-display').textContent = '0 - 100$';
    }
    
    if (params.skin_type) {
        document.getElementById('skin-type-display').textContent = params.skin_type;
    } else {
        document.getElementById('skin-type-display').textContent = 'All';
    }
    
    if (params.availability) {
        document.getElementById('availability-display').textContent = params.availability;
    } else {
        document.getElementById('availability-display').textContent = 'Available';
    }
}

// Get match probability badge class
function getMatchBadgeClass(probability) {
    if (probability >= 0.8) return 'excellent';
    if (probability >= 0.6) return 'good';
    return 'fair';
}

// Format probability as percentage
function formatProbability(probability) {
    return (probability * 100).toFixed(1) + '%';
}

// Search by URL
async function searchByURL() {
    const urlInput = document.getElementById('product-url');
    const url = urlInput.value.trim();
    const statusEl = document.getElementById('url-search-status');
    
    if (!url) {
        statusEl.className = 'url-status error';
        statusEl.textContent = 'Please enter a product URL';
        statusEl.style.display = 'block';
        return;
    }
    
    statusEl.className = 'url-status';
    statusEl.textContent = 'Searching...';
    statusEl.style.display = 'block';
    
    const loading = document.getElementById('loading');
    const grid = document.getElementById('products-grid');
    
    loading.style.display = 'block';
    grid.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/search-url`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusEl.className = 'url-status success';
            statusEl.textContent = `✓ Found ${data.total_found} matches`;
            displayProducts(data.matches || []);
        } else {
            statusEl.className = 'url-status error';
            statusEl.textContent = `✗ ${data.message}`;
            displayProducts([]);
        }
    } catch (error) {
        console.error('Error searching URL:', error);
        statusEl.className = 'url-status error';
        statusEl.textContent = '✗ Error searching URL. Please try again.';
        displayProducts([]);
    }
}

async function searchByProduct() {
  const brand = document.getElementById('brand-input').value.trim();
  const name = document.getElementById('name-input').value.trim();
  const statusEl = document.getElementById('search-status');
  const loading = document.getElementById('loading');
  const grid = document.getElementById('products-grid');

  if (!brand || !name) {
    statusEl.textContent = "Enter both brand and product name!";
    statusEl.style.color = '#dc3545';
    return;
  }
  
  statusEl.textContent = "Searching...";
  statusEl.style.color = '#6c757d';
  loading.style.display = 'block';
  grid.style.display = 'none';

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ brand, name })
    });
    const data = await response.json();
    if (data.success) {
      statusEl.textContent = `✓ Found ${data.results.length} matches`;
      statusEl.style.color = '#28a745';
      displayProducts(data.results);
    } else {
      statusEl.textContent = '✗ Error: ' + (data.error || 'Unknown error');
      statusEl.style.color = '#dc3545';
      displayProducts([]);
    }
  } catch (e) {
    console.error('Error:', e);
    statusEl.textContent = '✗ Server error. Please try again.';
    statusEl.style.color = '#dc3545';
    displayProducts([]);
  }
}

// Display products with probabilities
function displayProducts(products) {
    const grid = document.getElementById('products-grid');
    const loading = document.getElementById('loading');
    const noResults = document.getElementById('no-results');
    
    loading.style.display = 'none';
    
    if (!products || products.length === 0) {
        grid.style.display = 'none';
        noResults.style.display = 'block';
        document.getElementById('results-count').textContent = '0 products found';
        return;
    }
    
    noResults.style.display = 'none';
    grid.style.display = 'grid';
    document.getElementById('results-count').textContent = `${products.length} product${products.length !== 1 ? 's' : ''} found`;
    
    grid.innerHTML = products.map(product => {
        const probability = product.match_probability || product._similarity_score || 0;
        const accuracy = product.accuracy_percentage || (probability * 100);
        const badgeClass = getMatchBadgeClass(probability);
        const probabilityPercent = formatProbability(probability);
        const accuracyPercent = accuracy.toFixed(1) + '%';
        const stats = product.stats || {};
        const bestCategory = product.best_category || 'General';
        
        // Flags for category info
        const isCheapest = product.is_cheapest || false;
        const isMostAvailable = product.is_most_available || false;
        const isMostAccurate = product.is_most_accurate || false;
        
        // Product details
        const name = product.name || product.Name || 'Unknown Product';
        const brand = product.brand || product.Brand || 'Unknown Brand';
        const productType = product.product_type || product.type || product.Type || product.category || 'Unknown';
        const price = product.price || product.Price || product.cost || product.price_rsd || 0;
        const priceSign = product.price_sign || '';
        const currency = product.currency || 'RSD';
        const itemId = product.id || product.ID || 'N/A';
        
        // Handle colors
        let colorDisplay = '';
        if (product.product_colors && Array.isArray(product.product_colors) && product.product_colors.length > 0) {
            const colorNames = product.product_colors
                .map(c => c.colour_name || c.color_name || '')
                .filter(c => c)
                .slice(0, 3); // Show max 3 colors
            colorDisplay = colorNames.length > 0 
                ? colorNames.join(', ') 
                : `${product.product_colors.length} shades available`;
        } else {
            colorDisplay = product.color || product.Color || '';
        }
        
        // Product image
        const imageUrl = product.image_link || product.api_featured_image || '';
        const imageDisplay = imageUrl 
            ? `<img src="${imageUrl.startsWith('//') ? 'https:' + imageUrl : imageUrl}" alt="${name}" onerror="this.parentElement.innerHTML='💄'">`
            : '💄';
        
        // Stats display
        const statsHTML = `
            <div class="stats-section">
                <div class="stats-title">Product Stats</div>
                <div class="stat-item">
                    <span class="stat-label">Price Range</span>
                    <span class="stat-value">${stats.price_range || 'Unknown'}</span>
                </div>
                ${stats.color_count > 0 ? `
                <div class="stat-item">
                    <span class="stat-label">Color Options</span>
                    <span class="stat-value">${stats.color_count}</span>
                </div>
                ` : ''}
                ${stats.rating ? `
                <div class="stat-item">
                    <span class="stat-label">Rating</span>
                    <span class="stat-value">${stats.rating.toFixed(1)} ⭐</span>
                </div>
                ` : ''}
                <div class="stat-item">
                    <span class="stat-label">Brand Popularity</span>
                    <span class="stat-value">${stats.brand_popularity || 'Unknown'}</span>
                </div>
            </div>
        `;
        
        // Category info card
        const categoryInfoItems = [];
        if (isCheapest) categoryInfoItems.push('<span class="category-info-item cheapest">💰 Cheapest</span>');
        if (isMostAvailable) categoryInfoItems.push('<span class="category-info-item available">✅ Most Available</span>');
        if (isMostAccurate) categoryInfoItems.push('<span class="category-info-item accurate">🎯 Most Accurate</span>');
        
        const categoryInfoCard = categoryInfoItems.length > 0 
            ? `<div class="category-info-card">${categoryInfoItems.join('')}</div>` 
            : '';
        
        return `
            <div class="product-card">
                <div class="match-badge ${badgeClass}">
                    ${accuracyPercent} Accuracy
                </div>
                <div class="category-badge">
                    ${bestCategory}
                </div>
                ${categoryInfoCard}
                <div class="product-image">
                    ${imageDisplay}
                </div>
                <div class="product-info">
                    <div class="product-brand">${brand}</div>
                    <div class="product-name">${name}</div>
                    <div class="product-type">${productType}</div>
                    ${colorDisplay ? `<div class="product-colors">🎨 ${colorDisplay}</div>` : ''}
                    <div class="product-details">
                        <div class="product-price">${parseFloat(price).toFixed(2)} ${currency}</div>
                    </div>
                    ${statsHTML}
                    <div class="best-category">
                        <div class="best-category-label">Best For</div>
                        <div class="best-category-value">${bestCategory}</div>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill ${badgeClass}" style="width: ${accuracy}%"></div>
                    </div>
                    <div class="probability-text">Accuracy: ${accuracyPercent} | Match: ${probabilityPercent}</div>
                </div>
            </div>
        `;
    }).join('');
}

// Load matches from backend
async function loadMatches() {
    const params = getURLParams();
    const loading = document.getElementById('loading');
    const grid = document.getElementById('products-grid');
    
    loading.style.display = 'block';
    grid.style.display = 'none';
    
    try {
        // Build query string
        const queryParams = new URLSearchParams();
        if (params.price) queryParams.append('price', params.price);
        if (params.skin_type) queryParams.append('skin_type', params.skin_type);
        if (params.availability) queryParams.append('availability', params.availability);
        
        const url = `${API_BASE}/matches?${queryParams.toString()}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success) {
            displayProducts(data.matches || []);
        } else {
            console.error('Error loading matches:', data.message);
            displayProducts([]);
        }
    } catch (error) {
        console.error('Error loading matches:', error);
        displayProducts([]);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    displayFilters();
    
    // Always load matches (hardcoded 13 matches)
    loadMatches();
    
    // Allow Enter key in URL input
    const urlInput = document.getElementById('product-url');
    if (urlInput) {
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchByURL();
            }
        });
    }
});
