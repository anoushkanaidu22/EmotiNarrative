// Main JavaScript for EmotiNarrative

document.addEventListener('DOMContentLoaded', function() {
    //tab switching functionality
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                //deactivate all tabs
                tabs.forEach(t => t.classList.remove('active'));
                
                //hide all tab contents
                tabContents.forEach(content => content.style.display = 'none');
                
                //activate clicked tab
                tab.classList.add('active');
                
                //show corresponding content
                const targetId = tab.getAttribute('data-target');
                document.getElementById(targetId).style.display = 'block';
            });
        });
        
        //activate first tab by default
        tabs[0].click();
    }
    
    //range slider functionality
    const rangeSliders = document.querySelectorAll('.range-slider input');
    
    rangeSliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;
        
        //update value display on input
        slider.addEventListener('input', () => {
            valueDisplay.textContent = slider.value + '%';
        });
        
        //init value display
        valueDisplay.textContent = slider.value + '%';
    });
    
    //file input preview functionality
    const imageInput = document.getElementById('image_file');
    const imagePreview = document.getElementById('image_preview');
    
    if (imageInput && imagePreview) {
        imageInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
    
    //dashboard charts
    const createAccuracyChart = () => {
        const ctx = document.getElementById('accuracy-chart');
        
        if (!ctx) return;
        
        //fetch accuracy data from API
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                //process data for chart
                const textVersions = data.model_versions.filter(v => v.model_type === 'text');
                const imageVersions = data.model_versions.filter(v => v.model_type === 'image');
                
                const textDates = textVersions.map(v => new Date(v.created_date).toLocaleDateString());
                const textAccuracy = textVersions.map(v => v.accuracy * 100); // Convert to percentage
                
                const imageDates = imageVersions.map(v => new Date(v.created_date).toLocaleDateString());
                const imageAccuracy = imageVersions.map(v => v.accuracy * 100); // Convert to percentage
                
                //create chart
                const chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: textDates.length > imageDates.length ? textDates : imageDates,
                        datasets: [
                            {
                                label: 'Text Emotion Accuracy',
                                data: textAccuracy,
                                borderColor: '#ff99cc',
                                backgroundColor: 'rgba(255, 204, 230, 0.2)',
                                tension: 0.4
                            },
                            {
                                label: 'Image Emotion Accuracy',
                                data: imageAccuracy,
                                borderColor: '#17a2b8',
                                backgroundColor: 'rgba(23, 162, 184, 0.2)',
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Model Accuracy Improvement Over Time',
                                font: {
                                    size: 16
                                }
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                title: {
                                    display: true,
                                    text: 'Accuracy (%)'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Date'
                                }
                            }
                        }
                    }
                });
            })
            .catch(error => {
                console.error('Error fetching chart data:', error);
            });
    };
    
    //init charts if on dashboard page
    if (document.getElementById('accuracy-chart')) {
        createAccuracyChart();
    }
    
    //emotion visualization for text
    const visualizeTextEmotions = (emotions) => {
        const ctx = document.getElementById('text-emotions-chart');
        
        if (!ctx) return;
        
        const labels = Object.keys(emotions);
        const values = Object.values(emotions);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Emotion Score',
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.6)',
                        'rgba(54, 162, 235, 0.6)',
                        'rgba(255, 206, 86, 0.6)',
                        'rgba(75, 192, 192, 0.6)',
                        'rgba(153, 102, 255, 0.6)',
                        'rgba(255, 159, 64, 0.6)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        suggestedMax: 1.0
                    }
                }
            }
        });
    };
    
    //init text emotions chart if data available
    const emotionsData = document.getElementById('emotions-data');
    if (emotionsData) {
        try {
            const emotions = JSON.parse(emotionsData.textContent);
            visualizeTextEmotions(emotions);
        } catch (e) {
            console.error('Error parsing emotions data:', e);
        }
    }
});